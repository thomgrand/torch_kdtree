from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from .nn_distance import TorchKDTree, build_kd_tree


class BatchedKDTree:
    """Manages multiple TorchKDTree instances and executes batched KNN queries.

    Query API expects a tensor of shape [B, M, D] where B matches the number
    of trees in this batch.
    """

    def __init__(self, trees: List[TorchKDTree]) -> None:
        self.trees = list(trees)
        # Pre-allocate per-tree CUDA streams once so repeated .query() calls
        # pay no stream-creation overhead.
        self._streams: Optional[List["torch.cuda.Stream"]] = None
        if (self.trees
                and self.trees[0].device.type == 'cuda'
                and torch.cuda.is_available()):
            self._streams = [torch.cuda.Stream() for _ in self.trees]

    def __len__(self) -> int:
        return len(self.trees)

    def query(self, points_query: Tensor, nr_nns_searches: int = 1) -> Tuple[Tensor, Tensor]:
        """Query each tree with batched query points.

        Parameters
        ----------
        points_query : Tensor
            Query points of shape [B, M, D], where B == len(self.trees).
        nr_nns_searches : int
            Number of nearest neighbours to find.

        Returns
        -------
        tuple[Tensor, Tensor]
            (distances, indices) with shape [B, M, nr_nns_searches].

        Raises
        ------
        ValueError
            If the query shape does not match the batched tree layout.
        """
        if points_query.ndim != 3:
            raise ValueError(
                f"points_query must have shape [B, M, D], got {tuple(points_query.shape)}"
            )

        if points_query.shape[0] != len(self.trees):
            raise ValueError(
                f"Batch dimension ({points_query.shape[0]}) must match number of trees ({len(self.trees)})"
            )

        if not self.trees:
            b = points_query.shape[0]
            m = points_query.shape[1]
            dists = torch.empty([b, m, nr_nns_searches], dtype=points_query.dtype, device=points_query.device)
            inds = torch.empty([b, m, nr_nns_searches], dtype=torch.long, device=points_query.device)
            return dists, inds

        if self._streams is not None:
            return self._query_gpu(points_query, nr_nns_searches)
        else:
            return self._query_cpu(points_query, nr_nns_searches)

    def _query_gpu(self, points_query: Tensor, nr_nns_searches: int) -> Tuple[Tensor, Tensor]:
        """GPU path: dispatch all kernels concurrently, then per-stream sync + gather.

        Phase 1 — dispatch all N C++ kernels back-to-back on their respective
        pre-allocated streams.  The GPU hardware executes them concurrently.

        Phase 2 — iterate over trees in order: synchronize that tree's stream
        (CPU-blocks until THE KERNEL FOR THIS TREE is done), then do the
        shuffled_ind gather on the default stream.  While we wait on stream_i,
        streams i+1…N-1 keep running in GPU hardware.

        This correctly orders each gather after its own kernel (per-stream sync),
        avoids a global torch.cuda.synchronize(), and does not require the
        torch.cuda.stream() context manager (which can conflict with the
        py::gil_scoped_release inside the pybind11 binding).
        """
        # Phase 1 – dispatch all kernels concurrently
        first_tree = self.trees[0]
        points_query = points_query.to(first_tree.device)
        if not points_query.is_contiguous():
            points_query = points_query.contiguous()

        bsz, nr_query, _ = points_query.shape
        result_dists = torch.empty(
            [bsz, nr_query, nr_nns_searches],
            dtype=first_tree.dtype,
            device=first_tree.device,
        )
        result_idx = torch.empty(
            [bsz, nr_query, nr_nns_searches],
            dtype=first_tree.dtype_idx,
            device=first_tree.device,
        )

        for b, (tree, stream) in enumerate(zip(self.trees, self._streams)):
            query = points_query[b]
            if not query.is_contiguous():
                query = query.contiguous()
            tree.kdtree.query(
                query.data_ptr(), query.shape[0], nr_nns_searches,
                result_dists[b].data_ptr(), result_idx[b].data_ptr(),
                stream.cuda_stream,
            )

        # Phase 2 – per-stream sync then gather; later streams continue on GPU
        inds = torch.empty(
            [bsz, nr_query, nr_nns_searches],
            dtype=torch.long,
            device=first_tree.device,
        )
        dists = result_dists
        for b, (tree, stream) in enumerate(zip(self.trees, self._streams)):
            stream.synchronize()  # CPU-blocks only until THIS tree's kernel finishes
            query = points_query[b]
            inds[b] = tree.shuffled_ind[result_idx[b].long()]

            if (query.requires_grad or tree.ref_requires_grad) and torch.is_grad_enabled():
                dists[b] = torch.sum(
                    (query[:, None] - tree.points_ref_bak[inds[b]]) ** 2, dim=-1
                )
            if not tree.squared_distances:
                dists[b] = torch.sqrt(dists[b])

        return dists, inds

    def _query_cpu(self, points_query: Tensor, nr_nns_searches: int) -> Tuple[Tensor, Tensor]:
        """CPU path: sequential.

        Each tree.query() releases the GIL and runs its own OpenMP-parallel loop
        over query points, saturating available CPU cores.  Adding a Python-level
        thread pool on top creates GIL overhead and OpenMP thread contention,
        leading to a net slowdown rather than a speedup.
        """
        first_tree = self.trees[0]
        points_query = points_query.to(first_tree.device)
        if not points_query.is_contiguous():
            points_query = points_query.contiguous()

        bsz, nr_query, _ = points_query.shape
        dists = torch.empty(
            [bsz, nr_query, nr_nns_searches],
            dtype=first_tree.dtype,
            device=first_tree.device,
        )
        inds = torch.empty(
            [bsz, nr_query, nr_nns_searches],
            dtype=torch.long,
            device=first_tree.device,
        )
        for b, tree in enumerate(self.trees):
            d, i = tree.query(points_query[b], nr_nns_searches=nr_nns_searches)
            dists[b] = d
            inds[b] = i
        return dists, inds


def build_kd_tree_batched(
    points_ref_batch: Union[Tensor, np.ndarray],
    device: torch.device = None,
    squared_distances: bool = True,
    levels: int = None,
) -> BatchedKDTree:
    """Build a batched KD-Tree wrapper from [B, N, D] reference points.

    Parameters
    ----------
    points_ref_batch : Tensor | np.ndarray
        Reference point clouds with shape [B, N, D].
    device : torch.device | str | None
        Target device for query execution. Defaults to the tensor device.
    squared_distances : bool
        If True, return squared Euclidean distances.
    levels : int | None
        KD-Tree levels forwarded to each per-batch tree.
    """
    if isinstance(points_ref_batch, np.ndarray):
        points_ref_batch = torch.from_numpy(points_ref_batch)

    if not isinstance(points_ref_batch, torch.Tensor):
        raise TypeError("points_ref_batch must be a torch.Tensor or numpy.ndarray")

    if points_ref_batch.ndim != 3:
        raise ValueError(
            f"points_ref_batch must have shape [B, N, D], got {tuple(points_ref_batch.shape)}"
        )

    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = points_ref_batch.device

    trees: List[TorchKDTree] = []
    for b in range(points_ref_batch.shape[0]):
        trees.append(
            build_kd_tree(
                points_ref_batch[b],
                device=device,
                squared_distances=squared_distances,
                levels=levels,
            )
        )
    return BatchedKDTree(trees)

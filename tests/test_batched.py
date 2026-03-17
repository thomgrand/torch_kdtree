"""Tests for BatchedKDTree correctness, edge cases, and thread safety."""
import concurrent.futures

import pytest
import torch

from torch_kdtree import BatchedKDTree, build_kd_tree
from torch_kdtree.nn_distance import gpu_available


CUDA_TESTS_AVAILABLE = gpu_available and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_trees_and_queries(n_trees, n_points, d, dtype, device_str, k):
    """Build n_trees KD-Trees and corresponding query tensors."""
    torch_device = torch.device(device_str)
    trees = []
    queries = []
    for _ in range(n_trees):
        ref = torch.rand(n_points, d, dtype=dtype)
        trees.append(build_kd_tree(ref, device=torch_device))
        queries.append(torch.rand(50, d, dtype=dtype).to(torch_device))
    return trees, torch.stack(queries, dim=0)


# ---------------------------------------------------------------------------
# T008-1: Results match sequential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device_str", ["cpu"])
@pytest.mark.parametrize("n_trees", [2, 4])
def test_batched_results_match_sequential_cpu(d, dtype, device_str, n_trees):
    k = 5
    trees, queries = _make_trees_and_queries(n_trees, 200, d, dtype, device_str, k)

    batch = BatchedKDTree(trees)
    batch_dists, batch_inds = batch.query(queries, k)

    atol = 1e-5 if dtype == torch.float32 else 1e-9
    for i, tree in enumerate(trees):
        query = queries[i]
        seq_dists, seq_inds = tree.query(query, nr_nns_searches=k)
        assert torch.allclose(batch_dists[i], seq_dists, atol=atol), (
            f"Distances mismatch at tree {i}, d={d}, dtype={dtype}"
        )
        assert torch.equal(batch_inds[i], seq_inds), (
            f"Indices mismatch at tree {i}, d={d}, dtype={dtype}"
        )


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_trees", [2, 4])
@pytest.mark.skipif(not gpu_available, reason="GPU not available")
def test_batched_results_match_sequential_gpu(d, dtype, n_trees):
    k = 5
    device_str = "cuda"
    trees, queries = _make_trees_and_queries(n_trees, 200, d, dtype, device_str, k)

    batch = BatchedKDTree(trees)
    batch_dists, batch_inds = batch.query(queries, k)

    atol = 1e-5 if dtype == torch.float32 else 1e-9
    for i, tree in enumerate(trees):
        query = queries[i]
        seq_dists, seq_inds = tree.query(query, nr_nns_searches=k)
        assert torch.allclose(batch_dists[i], seq_dists, atol=atol), (
            f"GPU distances mismatch at tree {i}, d={d}, dtype={dtype}"
        )
        assert torch.equal(batch_inds[i], seq_inds), (
            f"GPU indices mismatch at tree {i}, d={d}, dtype={dtype}"
        )


# ---------------------------------------------------------------------------
# T008-2: Empty batch
# ---------------------------------------------------------------------------

def test_batched_empty_batch():
    batch = BatchedKDTree([])
    query = torch.empty((0, 0, 3), dtype=torch.float32)
    dists, inds = batch.query(query, nr_nns_searches=5)
    assert dists.shape == (0, 0, 5)
    assert inds.shape == (0, 0, 5)


# ---------------------------------------------------------------------------
# T008-3: Single-tree batch matches direct query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_single_tree(dtype):
    k = 3
    ref = torch.rand(100, 3, dtype=dtype)
    tree = build_kd_tree(ref, device=torch.device("cpu"))
    query = torch.rand(20, 3, dtype=dtype)

    batch = BatchedKDTree([tree])
    batch_dists, batch_inds = batch.query(query.unsqueeze(0), k)

    direct_dists, direct_inds = tree.query(query, nr_nns_searches=k)

    atol = 1e-5 if dtype == torch.float32 else 1e-9
    assert torch.allclose(batch_dists[0], direct_dists, atol=atol)
    assert torch.equal(batch_inds[0], direct_inds)


# ---------------------------------------------------------------------------
# T008-4: Mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_batched_mismatch_raises():
    ref = torch.rand(100, 3)
    tree = build_kd_tree(ref, device=torch.device("cpu"))
    batch = BatchedKDTree([tree, tree])
    query = torch.rand(10, 3)

    with pytest.raises(ValueError, match="Batch dimension"):
        batch.query(query.unsqueeze(0), nr_nns_searches=5)  # B=1 query, 2 trees → mismatch


# ---------------------------------------------------------------------------
# T008-5: CPU parallel — no race conditions
# ---------------------------------------------------------------------------

def test_batched_cpu_parallel():
    """Submit the same CPU batch query 4 times concurrently; all results must agree."""
    k = 5
    n_trees = 3
    dtype = torch.float32
    device = torch.device("cpu")

    ref_points = [torch.rand(200, 3, dtype=dtype) for _ in range(n_trees)]
    trees = [build_kd_tree(r, device=device) for r in ref_points]
    queries = torch.stack([torch.rand(30, 3, dtype=dtype) for _ in range(n_trees)], dim=0)

    batch = BatchedKDTree(trees)

    def _run(_):
        return batch.query(queries, k)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        all_results = list(executor.map(_run, range(4)))

    ref_dists, ref_inds = all_results[0]
    for run_dists, run_inds in all_results[1:]:
        for i in range(n_trees):
            assert torch.allclose(run_dists[i], ref_dists[i], atol=1e-6), (
                f"Race condition detected: distances differ at tree {i}"
            )
            assert torch.equal(run_inds[i], ref_inds[i]), (
                f"Race condition detected: indices differ at tree {i}"
            )

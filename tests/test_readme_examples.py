import numpy as np
import torch
from scipy.spatial import KDTree

from torch_kdtree import build_kd_tree, build_kd_tree_batched


def test_readme_single_tree_query_and_gradient_example():
    torch.manual_seed(0)
    np.random.seed(0)

    d = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points_ref = (torch.randn(size=(1000, d), dtype=torch.float32, device=device) * 1e3).requires_grad_()
    points_query = (torch.randn(size=(100, d), dtype=torch.float32, device=device) * 1e3).requires_grad_()

    torch_kdtree = build_kd_tree(points_ref)
    kdtree = KDTree(points_ref.detach().cpu().numpy())

    k = 5
    dists, inds = torch_kdtree.query(points_query, nr_nns_searches=k)
    dists_ref, inds_ref = kdtree.query(points_query.detach().cpu().numpy(), k=k)

    assert np.all(inds.detach().cpu().numpy() == inds_ref)
    assert np.allclose(torch.sqrt(dists).detach().cpu().numpy(), dists_ref, atol=1e-5)

    (0.5 * torch.sum(dists)).backward()
    grad_comp = torch.sum((points_query[:, None] - points_ref[inds]), dim=-2)
    assert points_query.grad is not None
    assert torch.allclose(points_query.grad, grad_comp)


def test_readme_batched_query_example():
    torch.manual_seed(0)
    np.random.seed(0)

    b, n, m, d = 8, 10000, 100, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points_ref_batched = torch.randn(b, n, d, dtype=torch.float32, device=device) * 1e3
    points_query_batched = torch.randn(b, m, d, dtype=torch.float32, device=device) * 1e3

    batched_kdtree = build_kd_tree_batched(points_ref_batched, device=device)
    ref_kdtrees = [KDTree(p.detach().cpu().numpy()) for p in points_ref_batched]

    k = 5
    dists, inds = batched_kdtree.query(points_query_batched, nr_nns_searches=k)
    dists_ref, inds_ref = zip(*[
        kdtree.query(p.detach().cpu().numpy(), k=k)
        for p, kdtree in zip(points_query_batched, ref_kdtrees)
    ])
    dists_ref = np.stack(dists_ref, axis=0)
    inds_ref = np.stack(inds_ref, axis=0)

    assert np.all(inds.detach().cpu().numpy() == inds_ref)
    assert np.allclose(torch.sqrt(dists).detach().cpu().numpy(), dists_ref, atol=1e-5)

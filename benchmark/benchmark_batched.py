"""Benchmark: sequential vs batched-tensor KD-Tree KNN queries."""
import argparse
import sys
import time

import torch

from torch_kdtree import build_kd_tree, build_kd_tree_batched
from torch_kdtree.nn_distance import gpu_available


def _time_sequential(trees, queries, k, repeats, sync_gpu=False):
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        for tree, q in zip(trees, queries):
            tree.query(q, nr_nns_searches=k)
        if sync_gpu:
            torch.cuda.synchronize()
        total += time.perf_counter() - t0
    return (total / repeats) * 1000.0  # ms


def _time_batched(batch, queries, k, repeats):
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        batch.query(queries, nr_nns_searches=k)
        total += time.perf_counter() - t0
    return (total / repeats) * 1000.0  # ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark batched vs sequential KD-Tree queries")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-trees", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--n-query", type=int, default=100,
                        help="query points per tree (lower = more SM headroom for concurrency)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dims", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available() or not gpu_available:
            print("CUDA is not available on this system. Exiting.")
            sys.exit(0)

    device = torch.device(args.device)
    dtype = torch.float32

    print(f"Building {args.n_trees} trees of {args.n_points} ref-pts each, {args.n_query} query-pts each (d={args.dims}, device={args.device})...")
    trees = []
    refs = []
    queries = []
    for _ in range(args.n_trees):
        ref = torch.rand(args.n_points, args.dims, dtype=dtype)
        refs.append(ref)
        trees.append(build_kd_tree(ref, device=device))
        queries.append(torch.rand(args.n_query, args.dims, dtype=dtype).to(device))

    queries = torch.stack(queries, dim=0)
    refs = torch.stack(refs, dim=0)
    batch = build_kd_tree_batched(refs, device=device)

    on_gpu = (args.device == "cuda")

    # Warm-up
    _time_sequential(trees, queries, args.k, 1, sync_gpu=on_gpu)
    _time_batched(batch, queries, args.k, 1)

    on_gpu = (args.device == "cuda")

    seq_ms = _time_sequential(trees, queries, args.k, args.repeats, sync_gpu=on_gpu)
    # Synchronize to flush any pending async GPU ops from sequential run
    # before dispatching batched kernels on non-default streams.
    if on_gpu:
        torch.cuda.synchronize()
    bat_ms = _time_batched(batch, queries, args.k, args.repeats)
    speedup = seq_ms / bat_ms if bat_ms > 0 else float("inf")

    col_w = [12, 10, 18, 14, 10]
    header = f"{'mode':<{col_w[0]}} {'n_trees':<{col_w[1]}} {'sequential_ms':<{col_w[2]}} {'batched_ms':<{col_w[3]}} {'speedup':<{col_w[4]}}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(
        f"{'sequential':<{col_w[0]}} {args.n_trees:<{col_w[1]}} {seq_ms:<{col_w[2]}.2f} {bat_ms:<{col_w[3]}.2f} {speedup:<{col_w[4]}.2f}"
    )
    print(sep)


if __name__ == "__main__":
    main()

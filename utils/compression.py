import numpy as np
import pandas as pd
import torch
from utils.DataLoader import Data


def random_compress_data(data: Data, ratio: float = 0.1, seed: int = 42) -> Data:
    """
    Randomly sample a subset of the data.
    :param data: Data object to compress
    :param ratio: float, ratio of data to keep (e.g., 0.1 for 10%)
    :param seed: int, random seed for reproducibility
    :return: Data object with sampled interactions
    """
    np.random.seed(seed)
    num_events = data.num_interactions
    k = int(ratio * num_events)

    # Random permutation and select first k indices
    idx = np.random.permutation(num_events)[:k]
    # Sort by timestamp to maintain temporal order
    idx = idx[np.argsort(data.node_interact_times[idx])]

    return Data(
        src_node_ids=data.src_node_ids[idx],
        dst_node_ids=data.dst_node_ids[idx],
        node_interact_times=data.node_interact_times[idx],
        edge_ids=data.edge_ids[idx],
        labels=data.labels[idx]
    )


def binary_search_window_compress_data(data: Data, ratio: float = 0.1, tol: float = 0.02,
                                       seed: int = 42, w_lo: float = 1.0, w_hi: float = 1e7,
                                       max_iter: int = 30, edge_features: np.ndarray = None) -> Data:
    """
    Per-(src,dst)-pair binary search over time window size to hit
    target compression ratio in [ratio-tol, ratio+tol].

    For each pair:
      - Binary search finds window w* such that ceil(pair_events / w*-bucket-count)
        lands in [target_lo, target_hi] events for that pair.
      - Events in each window are aggregated via delta-t weighted mean.
      - Representative timestamp = last event in window.

    Global result is concatenated across all pairs, then time-sorted.

    :param data: Data object to compress
    :param ratio: float, target compression ratio (e.g., 0.1 for 10%)
    :param tol: float, tolerance for compression ratio
    :param seed: int, random seed
    :param w_lo: float, minimum window size for binary search
    :param w_hi: float, maximum window size for binary search
    :param max_iter: int, maximum iterations for binary search
    :param edge_features: np.ndarray, edge features array (optional, for aggregation)
    :return: Data object with compressed interactions
    """
    print(f"--- Binary Search Window Compress (target: {ratio:.1%}) ---")

    # Create DataFrame for easier grouping
    df = pd.DataFrame({
        "src": data.src_node_ids,
        "dst": data.dst_node_ids,
        "t": data.node_interact_times,
        "label": data.labels,
        "edge_id": data.edge_ids,
        "idx": np.arange(len(data.src_node_ids)),
    }).sort_values(["src", "dst", "t"]).reset_index(drop=True)

    total_N = len(df)
    target_lo = ratio - tol
    target_hi = ratio + tol

    all_src, all_dst, all_t, all_label, all_edge_id = [], [], [], [], []

    pairs = df.groupby(["src", "dst"], sort=False)

    for (src_id, dst_id), grp in pairs:
        grp = grp.reset_index(drop=True)
        n_pair = len(grp)
        t_vals = grp["t"].values.astype(np.float64)

        target_n_lo = max(1, int(np.floor(n_pair * target_lo)))
        target_n_hi = max(1, int(np.ceil(n_pair * target_hi)))

        def compress_with_window(w):
            """Returns (n_compressed, t_rep)."""
            # Assign each event to a bucket: floor(t / w)
            buckets = np.floor(t_vals / w).astype(np.int64)
            _, inv = np.unique(buckets, return_inverse=True)

            n_buckets = int(inv.max()) + 1
            t_rep = np.zeros(n_buckets, dtype=np.float64)

            # Get representative timestamp (last event in each bucket)
            for i in range(n_pair):
                b = inv[i]
                t_rep[b] = t_vals[i]  # last event timestamp per bucket

            return n_buckets, t_rep

        # Binary search over window size
        lo, hi = float(w_lo), float(w_hi)
        best_w = hi
        converged = False

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            n_out, _ = compress_with_window(mid)

            if target_n_lo <= n_out <= target_n_hi:
                best_w = mid
                converged = True
                break
            elif n_out > target_n_hi:
                # Too many events → need larger window
                lo = mid
            else:
                # Too few events → need smaller window
                hi = mid

            if hi - lo < 1e-3:  # Numerical convergence
                best_w = mid
                break

        if not converged and n_pair == 1:
            best_w = w_lo  # Single-event pair: keep as-is

        n_out, t_rep = compress_with_window(best_w)

        # Use mode for labels, first edge_id in bucket
        all_src.append(np.full(n_out, src_id, dtype=np.int64))
        all_dst.append(np.full(n_out, dst_id, dtype=np.int64))
        all_t.append(t_rep)

        # For labels and edge_ids, we'll take the most common label and first edge_id per bucket
        buckets = np.floor(t_vals / best_w).astype(np.int64)
        _, inv = np.unique(buckets, return_inverse=True)
        n_buckets = int(inv.max()) + 1

        labels_out = np.zeros(n_buckets, dtype=data.labels.dtype)
        edge_ids_out = np.zeros(n_buckets, dtype=data.edge_ids.dtype)

        for i in range(n_pair):
            b = inv[i]
            if i == 0 or inv[i-1] != b:  # First event in bucket
                labels_out[b] = grp.iloc[i]["label"]
                edge_ids_out[b] = grp.iloc[i]["edge_id"]

        all_label.append(labels_out)
        all_edge_id.append(edge_ids_out)

    # Concatenate & sort by time
    all_src = np.concatenate(all_src)
    all_dst = np.concatenate(all_dst)
    all_t = np.concatenate(all_t)
    all_label = np.concatenate(all_label)
    all_edge_id = np.concatenate(all_edge_id)

    order = np.argsort(all_t, kind="stable")
    all_src = all_src[order]
    all_dst = all_dst[order]
    all_t = all_t[order]
    all_label = all_label[order]
    all_edge_id = all_edge_id[order]

    actual_ratio = len(all_src) / total_N
    print(f"   Input: {total_N} events | Output: {len(all_src)} events | "
          f"Actual ratio: {actual_ratio:.4f} (target: {target_lo:.2f}–{target_hi:.2f})")

    return Data(
        src_node_ids=all_src,
        dst_node_ids=all_dst,
        node_interact_times=all_t,
        edge_ids=all_edge_id,
        labels=all_label
    )

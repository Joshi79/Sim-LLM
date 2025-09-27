
import argparse
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd



def _zscore_matrix(X):
    """
    Z-score a 2D array [T, F] per feature (columns). If std==0, set that column to 0.
    """
    if X.size == 0:
        return X
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z


def _euclidean(a, b):
    return float(np.linalg.norm(a - b))


def dtw_distance(A, B, normalize= True):
    """
    Classic DTW distance between two sequences of shape [T, F].
    Per-timestep distance is Euclidean in feature space.
    """
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float("inf")

    # DP matrix with +inf borders
    dtw = np.full((n + 1, m + 1), np.inf, dtype=float)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = A[i - 1]
        for j in range(1, m + 1):
            cost = _euclidean(ai, B[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])
    dist = dtw[n, m]
    if normalize:
        dist /= (n + m)
    return float(dist)


def build_patient_series(df,id_col,time_col,feature_cols,sort_values= True,zscore_per_patient= True,fill_method= "ffill"):
    """
    Convert a long-form dataframe into a dict: patient_id -> array [T, F].
    """
    series = {}
    for pid, g in df.groupby(id_col):
        g = g.copy()
        if sort_values:
            g = g.sort_values(time_col)
        X = g[list(feature_cols)].astype(float)

        if fill_method == "ffill":
            X = X.ffill().bfill()
        elif fill_method == "bfill":
            X = X.bfill().ffill()
        elif fill_method == "zero":
            X = X.fillna(0.0)
        else:
            X = X.ffill().bfill()

        mat = X.values
        if zscore_per_patient:
            mat = _zscore_matrix(mat)

        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        series[str(pid)] = mat
    return series


def nearest_dtw_from_test_to_train(train_series,test_series,normalize= True):
    """
    For each test patient, compute DTW distance to each train patient and take the nearest one.
    """
    rows = []
    train_items = list(train_series.items())
    for test_id, test_seq in test_series.items():
        best_dist = math.inf
        best_train_id = None
        for train_id, train_seq in train_items:
            d = dtw_distance(test_seq, train_seq, normalize=normalize)
            if d < best_dist:
                best_dist = d
                best_train_id = train_id
        rows.append({
            "test_id": test_id,
            "nearest_train_id": best_train_id,
            "dtw_min": best_dist,
            "len_test": len(test_seq),
            "len_train": len(train_series[best_train_id]) if best_train_id is not None else None,
        })
    return pd.DataFrame(rows)


def pairwise_within_train_dtw(train_series,max_pairs= 200,normalize= True,random_state= 42):
    """
    Compute DTW for pairs within the train set (baseline). If combinations exceed max_pairs,
    sample pairs uniformly at random.
    """
    rng = np.random.default_rng(random_state)
    ids = list(train_series.keys())
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((ids[i], ids[j]))

    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    rows = []
    for a, b in pairs:
        d = dtw_distance(train_series[a], train_series[b], normalize=normalize)
        rows.append({"id_a": a, "id_b": b, "dtw": d,
                     "len_a": len(train_series[a]), "len_b": len(train_series[b])})
    return pd.DataFrame(rows)


def leakage_check(train_df, test_df, id_col):
    train_ids = set(map(str, train_df[id_col].unique()))
    test_ids = set(map(str, test_df[id_col].unique()))
    overlap = sorted(train_ids.intersection(test_ids))
    return (len(overlap) == 0), overlap


def summarize_distances(distances):
    if distances.empty:
        return {"count": 0}
    q = distances.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    return {
        "count": int(distances.shape[0]),
        "mean": float(distances.mean()),
        "std": float(distances.std(ddof=1)) if distances.shape[0] > 1 else 0.0,
        "min": float(distances.min()),
        "p50": float(q.get(0.5, np.nan)),
        "p90": float(q.get(0.9, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "max": float(distances.max()),
    }


def nearest_dtw_within_train(train_series, normalize=True, max_pairs=None, random_state=42):
    """
    For each training patient, compute DTW distance to all other training patients
    and find the nearest neighbor. This provides a baseline for comparison.
    """
    rng = np.random.default_rng(random_state)
    train_ids = list(train_series.keys())

    if len(train_ids) < 2:
        return pd.DataFrame(columns=["train_id", "nearest_train_id", "dtw_min",
                                     "len_train", "len_nearest"])

    if max_pairs is not None and len(train_ids) > max_pairs:
        sampled_ids = rng.choice(train_ids, size=max_pairs, replace=False)
        eval_ids = sampled_ids
    else:
        eval_ids = train_ids

    rows = []
    for train_id in eval_ids:
        train_seq = train_series[train_id]
        best_dist = math.inf
        best_neighbor_id = None

        for other_id in train_ids:
            if other_id == train_id:  # Skip self
                continue
            other_seq = train_series[other_id]
            d = dtw_distance(train_seq, other_seq, normalize=normalize)
            if d < best_dist:
                best_dist = d
                best_neighbor_id = other_id

        rows.append({
            "train_id": train_id,
            "nearest_train_id": best_neighbor_id,
            "dtw_min": best_dist,
            "len_train": len(train_seq),
            "len_nearest": len(train_series[best_neighbor_id]) if best_neighbor_id is not None else None,
        })

    return pd.DataFrame(rows)

def mean_pairwise_between_test_and_train(train_series, test_series, normalize=True):
    """
    Compute the mean DTW distance for all pairs between test and train sets
    without storing all individual distances.
    """
    total_distance = 0.0
    pair_count = 0
    for test_seq in test_series.values():
        for train_seq in train_series.values():
            total_distance += dtw_distance(test_seq, train_seq, normalize=normalize)
            pair_count += 1

    return total_distance / pair_count if pair_count > 0 else 0.0


def dtw_similarity_report(train_df, test_df, id_col, time_col, feature_cols,
                          normalize=True, max_within_pairs=200, zscore_per_patient=True,
                          max_train_neighbors=None):
    """
    Build the series, check leakage, compute nearest-neighbor DTW from test->train,
    and compute nearest-neighbor DTW within training set.
    """
    no_leakage, overlap = leakage_check(train_df, test_df, id_col)
    leakage_df = pd.DataFrame([{"no_leakage": no_leakage, "overlap_ids": overlap}])

    if not no_leakage:
        pass

    train_series = build_patient_series(
        train_df, id_col=id_col, time_col=time_col, feature_cols=feature_cols,
        zscore_per_patient=zscore_per_patient
    )
    test_series = build_patient_series(
        test_df, id_col=id_col, time_col=time_col, feature_cols=feature_cols,
        zscore_per_patient=zscore_per_patient
    )

    # Test to train nearest neighbors
    nn_df = nearest_dtw_from_test_to_train(train_series, test_series, normalize=normalize)

    # Train to train nearest neighbors
    train_nn_df = nearest_dtw_within_train(train_series, normalize=normalize,
                                           max_pairs=max_train_neighbors)

    # Pairwise within train (for additional analysis)
    if len(train_series) >= 2:
        baseline_df = pairwise_within_train_dtw(
            train_series, max_pairs=max_within_pairs, normalize=normalize
        )
    else:
        baseline_df = pd.DataFrame(columns=["id_a", "id_b", "dtw", "len_a", "len_b"])

    return leakage_df, nn_df, train_nn_df, baseline_df



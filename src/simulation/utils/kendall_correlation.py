import pandas as pd
import numpy as np
from scipy import signal
from itertools import combinations
from pathlib import Path


def create_trend_analysis_summary(file_path: str) -> str:
    """
    Read a CSV, extract linear trends,
    """
    # ---- Hard-coded inputs ----
    df = pd.read_csv(file_path)
    id_col = "user_id"
    feature_cols = [
        'numberRating', 'highestRating', 'lowestRating',
        'medianRating', 'sdRating',
        'numberLowRating', 'numberMediumRating', 'numberHighRating',
        'numberMessageReceived', 'numberMessageRead', 'readAllMessage'
    ]

    # Keep only features present in the data
    features = [c for c in feature_cols if c in df.columns]
    if not features or id_col not in df.columns:
        return f"Linear-trend Kendall τ | file={Path(file_path).name}"

    # Friendly display names
    nice = {
        'numberRating': "number of ratings",
        'highestRating': "highest rating",
        'lowestRating': "lowest rating",
        'medianRating': "median rating",
        'sdRating': "rating variability (SD)",
        'numberLowRating': "low ratings (1–2)",
        'numberMediumRating': "medium ratings (3–5)",
        'numberHighRating': "high ratings (6–7)",
        'numberMessageReceived': "messages received",
        'numberMessageRead': "messages read",
        'readAllMessage': "read-all indicator",
    }

    def strength_label(t: float) -> str:
        if not np.isfinite(t): return "n/a"
        a = abs(t)
        if a < 0.10: return "negligible"
        if a < 0.20: return "very weak"
        if a < 0.35: return "weak"
        if a < 0.50: return "moderate"
        if a < 0.70: return "strong"
        return "very strong"

    # Collect pairwise Kendall τ across patients (on linear trends)
    pair_vals = {tuple(sorted(p)): [] for p in combinations(features, 2)}
    var_patient_counts = {f: 0 for f in features}

    for _, g in df.groupby(id_col):
        if len(g) < 3:
            continue
        g = g.sort_index()

        trends = {}
        for f in features:
            s = g[f].astype(float).interpolate(limit_direction="both")
            if s.count() < 3:
                continue
            cyc = signal.detrend(s.to_numpy(dtype=float))
            tr = s.to_numpy(dtype=float) - cyc
            tr = pd.Series(tr, index=s.index)
            if tr.std() <= 1e-9:
                continue
            trends[f] = tr

        if not trends:
            continue

        for f in trends.keys():
            var_patient_counts[f] += 1

        avail = sorted(trends.keys())
        for a, b in combinations(avail, 2):
            try:
                tau = trends[a].corr(trends[b], method="kendall")
                if np.isfinite(tau):
                    pair_vals[(a, b)].append(float(tau))
            except Exception:
                pass

    # Average τ per pair across patients
    pair_avg = {}
    pair_n = {}
    for pair, vals in pair_vals.items():
        if vals:
            pair_avg[pair] = float(np.mean(vals))
            pair_n[pair] = int(len(vals))

    # Build symmetric matrix
    mat = pd.DataFrame(np.nan, index=features, columns=features, dtype=float)
    for (a, b), v in pair_avg.items():
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    np.fill_diagonal(mat.values, 1.0)
    mean_with_others = mat.replace(1.0, np.nan).mean(axis=1)

    lines = []
    lines.append(
        f"Linear-trend Kendall τ (within-patient) | file={Path(file_path).name} "
        f"| users={df[id_col].nunique()} | features={len(features)}"
    )

    if pair_avg:
        all_pairs = [(a, b, pair_avg[(a, b)], pair_n[(a, b)]) for (a, b) in pair_avg.keys()]
        all_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        lines.append("Top patterns:")
        for a, b, tau, n in all_pairs[:8]:
            relation = "More {} → more {}".format(nice.get(a, a), nice.get(b, b)) if tau > 0 \
                else "More {} → fewer {}".format(nice.get(a, a), nice.get(b, b))
            lines.append(f"- {relation} (τ={tau:+.3f}, n={n})")

    lines.append("Per-variable summary:")
    for f in features:
        row = mat.loc[f].drop(labels=[f], errors="ignore")
        pos = row[row > 0].idxmax() if (row > 0).any() else None
        neg = row[row < 0].idxmin() if (row < 0).any() else None
        pos_v = (row[pos] if pos else np.nan)
        neg_v = (row[neg] if neg else np.nan)
        mean_tau = mean_with_others.get(f, np.nan)
        parts = [
            f"{nice.get(f, f)} (n={var_patient_counts.get(f, 0)})",
            f"mean τ={mean_tau:+.3f} ({strength_label(mean_tau)})"
        ]
        if pos:
            parts.append(f"+{nice.get(pos, pos)} {pos_v:+.3f}")
        if neg:
            parts.append(f"−{nice.get(neg, neg)} {neg_v:+.3f}")
        lines.append(" | ".join(parts))

    overall = np.nanmean(mean_with_others.to_numpy()) if len(mean_with_others) else np.nan
    lines.append(f"Overall mean τ={overall:+.3f} ({strength_label(overall)})")
    lines.append("Read: τ>0 → variables trend up/down together; τ<0 → they move oppositely. Correlation ≠ causation.")

    return "\n".join(lines)


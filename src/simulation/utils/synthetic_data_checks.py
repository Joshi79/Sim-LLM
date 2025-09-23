import pandas as pd
import numpy as np


def simulation_statistics(df) -> str:
    """
    Compute concise descriptive stats per relevant column and return a natural-language summary.
    Works on any DataFrame-like input with array-like column access.
    """
    cols = [
        "numberRating", "highestRating", "lowestRating", "medianRating",
        "sdRating", "numberLowRating", "numberMediumRating",
        "numberHighRating", "numberMessageReceived", "numberMessageRead"
    ]
    parts = []

    if df is None or (hasattr(df, 'shape') and df.shape[0] == 0):
        return "No data available for temporal check"

    for col in cols:
        if not hasattr(df, 'columns') or col not in df.columns:
            continue

        series = df[col]
        non_null = series.dropna()

        if non_null.empty:
            parts.append(f"Column '{col}': no valid values.")
            continue

        # Basic stats
        min_val = non_null.min()
        max_val = non_null.max()
        median_val = non_null.median()
        std_val = non_null.std()

        # Value counts and modes
        value_counts = non_null.value_counts().sort_index()
        mode_vals = list(value_counts[value_counts == value_counts.max()].index.astype(int))

        # Quartiles for IQR
        q1 = non_null.quantile(0.25)
        q2 = non_null.quantile(0.5)
        q3 = non_null.quantile(0.75)

        # Formatting counts
        top_str = ", ".join([f"{val}: {cnt}" for val, cnt in value_counts.items()])
        mode_str = ", ".join(map(str, mode_vals))

        summary = (
            f"Column '{col}': median={median_val:.2f}, std={std_val:.2f}, "
            f"range=[{min_val}, {max_val}], Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}, "
            f"mode(s)={mode_str}. Value Counts: {top_str}."
        )
        parts.append(summary)

    return "\n".join(parts)



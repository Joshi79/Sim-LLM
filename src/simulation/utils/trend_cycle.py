import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
from collections import defaultdict


def create_trend_analysis_summary(file_path):
    """
    Create a complete trend and cycle analysis summary
    """
    df = pd.read_csv(file_path)
    id_col = "user_id"

    feature_cols = [
        'numberRating', 'highestRating', 'lowestRating',
        'medianRating', 'sdRating',
        'numberLowRating', 'numberMediumRating', 'numberHighRating',
        'numberMessageReceived', 'numberMessageRead',
    ]

    trend_stats = defaultdict(list)
    cycle_stats = defaultdict(list)

    # Process each user separately
    for user_id, user_data in df.groupby(id_col):
        user_data = user_data.reset_index(drop=True)

        for feature in feature_cols:
            if feature not in user_data.columns:
                continue

            values = user_data[feature].to_numpy(dtype=float)

            if np.isnan(values).any():
                values = pd.Series(values).ffill().bfill().fillna(0.0).to_numpy()

            if len(values) < 3:
                continue

            # Decompose into trend and cycle
            cycle_component = signal.detrend(values)
            trend_component = values - cycle_component

            # Calculate trend slope
            x = np.arange(len(values))
            coeffs = np.polyfit(x, trend_component, 1)
            slope = coeffs[0]

            mean_val = np.mean(values)
            relative_slope = slope / mean_val if mean_val != 0 else 0

            # Calculate cycle characteristics
            cycle_std = np.std(cycle_component)
            cycle_range = np.max(cycle_component) - np.min(cycle_component)

            relative_cycle_std = cycle_std / abs(mean_val) if mean_val != 0 else 0
            relative_cycle_range = cycle_range / abs(mean_val) if mean_val != 0 else 0

            trend_stats[feature].append(relative_slope)
            cycle_stats[feature].append({
                'std': relative_cycle_std,
                'range': relative_cycle_range,
                'raw_std': cycle_std
            })

    # Aggregate statistics
    feature_trends = {}
    feature_cycles = {}

    for feature, slopes in trend_stats.items():
        if slopes:
            feature_trends[feature] = {'avg_trend': float(np.mean(slopes))}

    for feature, cycles in cycle_stats.items():
        if cycles:
            feature_cycles[feature] = {'avg_volatility': float(np.mean([c['std'] for c in cycles]))}

    # Generate summary
    lines = []
    lines.append("## Trend and Cycle Summary")
    lines.append("-" * 30)

    all_features = set(feature_trends.keys()) | set(feature_cycles.keys())

    for feature in sorted(all_features):
        trend = feature_trends.get(feature, {}).get('avg_trend', 0)
        vol = feature_cycles.get(feature, {}).get('avg_volatility', 0)

        if trend > 0.01:
            trend_symbol = "↑"
        elif trend < -0.01:
            trend_symbol = "↓"
        else:
            trend_symbol = "→"

        lines.append(f"{feature}: {trend_symbol}  (trend={trend:+.3f}, vol={vol:.2f})")

    all_trends = [stats['avg_trend'] for stats in feature_trends.values()]
    all_vols = [stats['avg_volatility'] for stats in feature_cycles.values()]

    guidelines = """
      ### Simulation Guidelines
      ------------------------------
      When generating synthetic data:

      **For Trends:**
      - Apply gradual changes according to trend direction
      - Trend value indicates rate of change per time unit
      - Positive = increase over time, Negative = decrease

      **For Cycles/Volatility:**
      - Add short-term fluctuations based on volatility level
      - High volatility = larger daily/weekly swings
      - Low volatility = minimal variation around trend

      **Combined Patterns:**
      - Rising + Volatile: Upward trend with significant fluctuations
      - Rising + Smooth: Steady, predictable increase
      - Stable + Volatile: Oscillates around fixed mean
      - Stable + Smooth: Nearly constant values
      """

    lines.append(f"\nOverall: Trend={np.mean(all_trends):+.3f}, Volatility={np.mean(all_vols):.3f}")
    lines.append(guidelines)

    return "\n".join(lines)


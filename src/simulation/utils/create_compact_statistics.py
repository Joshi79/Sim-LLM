import pandas as pd
import numpy as np
import io


# can be used for local testing
#FILE_PATH =



def create_compact_statistics_summary(file_path: str):
    """
    Loads the CSV, computes concise descriptive stats per relevant column, and returns a natural-language summary.
    """
    df = pd.read_csv(file_path)
    cols = [
        "numberRating", "highestRating", "lowestRating", "medianRating",
        "sdRating", "numberLowRating", "numberMediumRating",
        "numberHighRating", "numberMessageRead"
    ]
    parts = []
  
    for col in cols:
        if col not in df.columns:
            continue

        series = df[col]
        non_null = series.dropna()


        # Basic stats
        min_val = non_null.min()
        max_val = non_null.max()
        median_val = non_null.median()
        std_val = non_null.std()

        # Value counts and modes
        value_counts = non_null.value_counts().sort_index()
        mode_vals = list(value_counts[value_counts == value_counts.max()].index.astype(int))

        # Percentiles give all percentiles
        q1 = non_null.quantile(0.25)
        q2 = non_null.quantile(0.5)
        q3 = non_null.quantile(0.75)


        # Formatting
        top_str = ", ".join([f"{float(val)}: {cnt}" for val, cnt in value_counts.items()])
        mode_str = ", ".join(map(str, mode_vals))

        summary = (
            f"Column '{col}': "
            f"median={median_val:.2f}, std={std_val:.2f}, "
            f"range=[{min_val}, {max_val}], Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}, "
            f"mode(s)={mode_str}. Value Counts: {top_str}."
        )
        parts.append(summary)

    return "\n".join(parts)


#if __name__ == "__main__":
    # can be used for local testing
    #FILE_PATH =
    #summary = create_compact_statistics_summary(FILE_PATH)
    # Print the summary
    #print(summary)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np


def visualize_distributions_two_datasets(
        synthetic_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_path: str | None = None,
        labels: dict | None = None,
        continuous_cols: list | None = None,
        show: bool = False
):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cols = [
        'numberRating', 'highestRating', 'lowestRating', 'medianRating', 'sdRating',
        'numberLowRating', 'numberMediumRating', 'numberHighRating',
        'numberMessageRead', 'readAllMessage',
    ]
    labels = labels or {}
    if continuous_cols is None:
        continuous_cols = ['sdRating']

    colors = {
        "Synthetic": "#009E73",
        "Test": "#D55E00"
    }

    fig, axes = plt.subplots(2, 5, figsize=(42, 22))
    axes = axes.flatten()
    plot_idx = 0

    for col in cols:
        if (col not in synthetic_df.columns or col not in test_df.columns):
            axes[plot_idx].set_visible(False)
            plot_idx += 1
            continue

        ax = axes[plot_idx]
        plot_idx += 1

        if col in continuous_cols:
            df_plot = pd.DataFrame({
                'Synthetic': pd.to_numeric(synthetic_df[col], errors='coerce').dropna(),
                'Test': pd.to_numeric(test_df[col], errors='coerce').dropna()
            }).melt(var_name='Type', value_name='Value')

            sns.kdeplot(
                data=df_plot, x='Value', hue='Type',
                fill=True, alpha=0.4, linewidth=2, common_norm=False,
                warn_singular=False, legend=False, palette=colors, ax=ax
            )
            ax.set_ylabel('Density', fontsize=40, fontweight='bold')

        elif col == "medianRating":

            datasets = {
                "Synthetic": synthetic_df,
                "Test": test_df
            }

            domain = np.arange(0.0, 7.0 + 0.5, 0.5)

            def snap(series):
                s = pd.to_numeric(series, errors="coerce").dropna()
                return (np.round(s * 2) / 2)

            proportions = {}
            for name, df_ in datasets.items():
                s = snap(df_[col])
                vc = pd.Series(s).value_counts()
                n = len(s)
                proportions[name] = [(vc.get(v, 0) / n) if n > 0 else 0 for v in domain]

            x = domain.astype(float)
            n_groups = len(datasets)
            bar_width = 0.2
            offsets = np.linspace(-bar_width / 2, bar_width / 2, n_groups)

            for (name, props), off in zip(proportions.items(), offsets):
                ax.bar(x + off, props, bar_width, label=name, color=colors[name], alpha=0.5)


            tick_positions = [0, 1, 2, 3, 4, 5, 6, 7]
            tick_labels = ['0', '1', '2', '3', '4', '5', '6', '7']

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=28, fontweight='bold')
            ax.set_xlim(-0.25, 7.25)
            ax.set_ylabel('Proportion', fontsize=35, fontweight='bold')

        else:
            datasets = {
                "Synthetic": synthetic_df,
                "Test": test_df
            }

            all_values = set()
            for df_ in datasets.values():
                all_values.update(pd.to_numeric(df_[col], errors="coerce").dropna().unique())
            domain = sorted(all_values)

            def snap(series):
                return pd.to_numeric(series, errors="coerce").dropna()

            proportions = {}
            for name, df_ in datasets.items():
                s = snap(df_[col])
                vc = pd.Series(s).value_counts()
                n = len(s)
                proportions[name] = [(vc.get(v, 0) / n) if n > 0 else 0 for v in domain]

            x = np.arange(len(domain))
            n_groups = len(datasets)
            total_width = 0.6  # Adjusted for two bars
            bar_width = total_width / n_groups
            start = -total_width / 2 + bar_width / 2

            for i, (name, props) in enumerate(proportions.items()):
                ax.bar(x + start + i * bar_width, props, bar_width,
                       label=name, color=colors[name], alpha=0.5)

            def fmt(v):
                return str(int(v)) if float(v).is_integer() else f"{v:.1f}"

            ax.set_xticks(x)
            ax.set_xticklabels([fmt(v) for v in domain], ha="center")
            ax.set_xlim(x[0] - total_width / 2 if len(x) > 0 else -0.5,
                        x[-1] + total_width / 2 if len(x) > 0 else 0.5)
            ax.set_ylabel('Proportion', fontsize=35, fontweight='bold')

        ax.set_xlabel(labels.get(col, col), fontsize=40, fontweight='bold', labelpad=10)
        ax.tick_params(labelsize=30)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.grid(False)
        ax.set_axisbelow(True)

    for j in range(plot_idx, 10):
        axes[j].set_visible(False)

    # Make legend bold
    legend = axes[0].legend(fontsize=28, loc='upper right', framealpha=0.95)
    # Set font weight to bold for legend text
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout(pad=3.0)

    # Black borders around bars
    for ax in axes:
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', pad_inches=0.2,
                    format='pdf', facecolor='white', edgecolor='none')
    else:
        plt.savefig(f"../../reports/output_data_fidelity/distributions_synthetic_vs_test_{current_time}.pdf",
                    dpi=1200, bbox_inches='tight', pad_inches=0.2,
                    format='pdf', facecolor='white', edgecolor='none')

    if show:
        plt.show()

    plt.close(fig)


# Example run
if __name__ == "__main__":
    synthetic_path = r"../../data/synthetic_data_baseline_prompt.csv"
    real_test_path = r"../../data/test_df.csv"
    syntehtic_data_no_dataset_info = r"../../data/prompt_no_info_orignal_data.csv"
    synthetic_data_no_grounding = r"../../data/prompt_not_grounded_in_synthetic.csv"

    synthetic_df_no_info = pd.read_csv(syntehtic_data_no_dataset_info)
    synthetic_df_no_grounding = pd.read_csv(synthetic_data_no_grounding)
    synthetic_df = pd.read_csv(synthetic_path)
    test_df = pd.read_csv(real_test_path)

    pretty = {
        'numberRating': 'Number of Ratings',
        'highestRating': 'Highest Rating',
        'lowestRating': 'Lowest Rating',
        'medianRating': 'Median Rating',
        'sdRating': 'SD Rating',
        'numberLowRating': 'Number Low Ratings',
        'numberMediumRating': 'Number Medium Ratings',
        'numberHighRating': 'Number High Ratings',
        'numberMessageRead': 'Number Messages Read',
        'readAllMessage': 'Read All Messages'
    }

    visualize_distributions_two_datasets(
        synthetic_df, test_df,
        labels=pretty,
        continuous_cols=['sdRating'],
        show=False
    )
    print("Done! Check for the PDF output.")
    visualize_distributions_two_datasets(
        synthetic_df_no_info, test_df,
        labels=pretty,
        continuous_cols=['sdRating'],
        show=False
    )
    print("Done! Check for the PDF output.")

    visualize_distributions_two_datasets(
        synthetic_df_no_grounding, test_df,
        labels=pretty,
        continuous_cols=['sdRating'],
        show=False
    )
    print("Done! Check for the PDF output.")
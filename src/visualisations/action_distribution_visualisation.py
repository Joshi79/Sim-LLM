import matplotlib.pyplot as plt
import numpy as np
import json
import datetime


# needed for
file_path = r"../../reports/output_policy_utility/boostrapped_results_fixed_learned_policies.json"


with open(file_path, 'r') as f:
    data = json.load(f)

original_vals = data["policies"]["original_30"]["action_distribution"]
synthetic_vals_60 = data["policies"]["synthetic_60"]["action_distribution"]
synthetic_vals_100 = data["policies"]["synthetic_100"]["action_distribution"]
merged_vals = data["policies"]["merged_120"]["action_distribution"]



# Prepare data for all four datasets
action_types = list(original_vals.keys())
original_counts = list(original_vals.values())
synthetic_60_counts = list(synthetic_vals_60.values())
synthetic_100_counts = list(synthetic_vals_100.values())
merged_counts = list(merged_vals.values())

# Calculate proportions for all datasets
total_orig = sum(original_counts)
total_syn_60 = sum(synthetic_60_counts)
total_syn_100 = sum(synthetic_100_counts)
total_merged = sum(merged_counts)

original_proportions = [count/total_orig for count in original_counts]
synthetic_60_proportions = [count/total_syn_60 for count in synthetic_60_counts]
synthetic_100_proportions = [count/total_syn_100 for count in synthetic_100_counts]
merged_proportions = [count/total_merged for count in merged_counts]

# Set up the data structure for plotting
algorithms = ['Original data', 'Synthetic data \n(n=60)', 'Synthetic data \n(n=100)', 'Merged data']
data = {
    'No message': [original_proportions[0], synthetic_60_proportions[0], synthetic_100_proportions[0], merged_proportions[0]],
    'Encouraging': [original_proportions[1], synthetic_60_proportions[1], synthetic_100_proportions[1], merged_proportions[1]],
    'Informing': [original_proportions[2], synthetic_60_proportions[2], synthetic_100_proportions[2], merged_proportions[2]],
    'Affirming': [original_proportions[3], synthetic_60_proportions[3], synthetic_100_proportions[3], merged_proportions[3]]
}
"""
colors = {
    'No message': '#d62728',     # Red
    'Encouraging': '#ff7f0e',    # Orange
    'Informing': '#2ca02c',      # Green
    'Affirming': '#1f77b4'       # Blue
}
"""
colors = {
    'No message':   '#ff7f0e',  # Deep Purple
    'Encouraging':  '#31688E',  # Teal-Blue
    'Informing':    '#35B779',  # Green
    'Affirming':    '#FDE725',  # Bright Yellow
}
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(algorithms))
width = 0.15
multiplier = 0

# Create bars for each action type
for attribute, measurement in data.items():
    offset = width * multiplier
    bars = ax.bar(x + offset, measurement, width, label=attribute,
                  color=colors[attribute], edgecolor='black', linewidth=0.5)
    multiplier += 1

# Customize tick labels
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

for label in ax.get_xticklabels():
    label.set_fontsize(22)
    label.set_fontweight('bold')

for label in ax.get_yticklabels():
    label.set_fontsize(22)
    label.set_fontweight('bold')

ax.set_xlabel('Datasets', fontsize=24, fontweight='bold')
ax.set_ylabel('Message Proportion', fontsize=24, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(algorithms, rotation=0)
ax.set_ylim(0, 0.8)

# Add legend
bold_legend = {'weight': 'bold', 'size': 16}

legend = ax.legend(
    loc='upper left',
    frameon=True,
    fancybox=False,
    shadow=False,
    prop=bold_legend,  # size lives here
    markerscale=1.6,
    handlelength=1.2,
    handletextpad=0.6,
    borderpad=0.4
)
legend.get_frame().set_linewidth(1.2)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"../../reports/output_policy_utility/action_distribution.pdf",
           dpi=1200, bbox_inches='tight', pad_inches=0.2,
           format='pdf', facecolor='white', edgecolor='none')
print("Action Distribution comparison saved as 'action_distribution_all_datasets.pdf'")


# Print summary for all datasets
datasets_info = [
    ("Original Data", original_counts, original_proportions, total_orig),
    ("Synthetic 60P", synthetic_60_counts, synthetic_60_proportions, total_syn_60),
    ("Synthetic 100P", synthetic_100_counts, synthetic_100_proportions, total_syn_100),
    ("Merged Data", merged_counts, merged_proportions, total_merged)
]

print(f"\nAction Distribution Comparison:")
for dataset_name, counts, proportions, total in datasets_info:
    print(f"\n{dataset_name} (Total: {total:,}):")
    for action, count, prop in zip(action_types, counts, proportions):
        print(f"  {action}: {count:,} ({prop:.3%})")


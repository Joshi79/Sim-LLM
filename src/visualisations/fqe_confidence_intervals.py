import matplotlib.pyplot as plt
import json
import numpy as np
import datetime


path_unified = r"C:\Users\User\PycharmProjects\master_thesis\results\unified_bootstrap_results_20250824_223429.json"

# time stemp is needed to avoid overwriting files
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    with open(path_unified, 'r') as f:
        unified_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {path_unified}")
    raise SystemExit(1)

# Extract all policies from the data
policies_data = unified_data.get('policies', {})

labels_raw = []
point_estimates = []
medians = []
ci_lower_bounds = []
ci_upper_bounds = []

policy_order = [
    'original_30', 'synthetic_60', 'synthetic_100', 'merged_120',
    'fixed_action_0', 'fixed_action_1', 'fixed_action_2', 'fixed_action_3'
]

print("--- 95% Confidence Intervals (Error Percentile) ---")
for policy_key in policy_order:
    policy = policies_data.get(policy_key)
    if not policy:
        print(f"Warning: Policy '{policy_key}' not found in JSON data. Skipping.")
        continue

    description = policy.get('description', policy_key)
    labels_raw.append(description)

    pe = policy.get('point_estimate')
    point_estimates.append(np.nan if pe is None else pe)

    med = policy.get('bootstrap_median')
    medians.append(np.nan if med is None else med)

    # Extract the 'error_percentile' confidence interval
    ci = policy.get('confidence_intervals', {}).get('error_percentile', {}).get('ci')
    if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        ci_lower_bounds.append(ci[0])
        ci_upper_bounds.append(ci[1])
        print(f"{description:<25}: [{ci[0]:.3f}, {ci[1]:.3f}]")
    else:
        ci_lower_bounds.append(np.nan)
        ci_upper_bounds.append(np.nan)
        print(f"Warning: Confidence interval not found for policy '{policy_key}'.")



# Plot the FQE confidence Intervals
fig, ax = plt.subplots(figsize=(16, 9))


### Calculate the error confidence intervals
pe_arr = np.array(point_estimates, dtype=float)
ci_lo = np.array(ci_lower_bounds, dtype=float)
ci_hi = np.array(ci_upper_bounds, dtype=float)

lower_error = pe_arr - ci_lo
upper_error = ci_hi - pe_arr
asymmetric_error = [lower_error, upper_error]

x_pos = np.arange(len(pe_arr))

labels_compact = [
    "Original",
    "Synthetic\n n=60",
    "Synthetic\n n=100",
    "Merged",
    "No\n message",
    "Encouraging",
    "Informing",
    "Affirming"
]


labels_compact = labels_compact[:len(x_pos)]

# Error bars (no connecting line)
ax.errorbar(
    x_pos, pe_arr, yerr=asymmetric_error, fmt='none',
    ecolor='black', capsize=7, elinewidth=1.5, capthick=1.5, zorder=3
)

# Point Estimate and Median markers
ax.scatter(x_pos, pe_arr,  marker='D', zorder=5, s=70, label='Point Estimate')
ax.scatter(x_pos, np.array(medians, dtype=float),marker='_', zorder=4, s=250, linewidth=3, label='Median')

ax.set_ylabel('FQE value', fontsize=20, fontweight='bold', labelpad=10)
ax.set_xlabel('Policy',    fontsize=20, fontweight='bold', labelpad=8)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_compact, rotation=0, ha='center')

# Tick params: make bold and size-tuned (smaller x to save space)
ax.tick_params(axis='x', which='major', labelsize=20, width=1.5)
ax.tick_params(axis='y', which='major', labelsize=20, width=1.5)

for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

ax.axhline(0, linewidth=0.8, linestyle='--', color='black')

ax.yaxis.grid(True, alpha=0.4)

bold_legend = {'weight': 'bold', 'size': 14}
legend = ax.legend(
    loc='upper left',
    framealpha=0.95,
    fancybox=True,
    prop=bold_legend,
    markerscale=1.6,
    handlelength=1.2,
    handletextpad=0.6,
    borderpad=0.4
)
legend.get_frame().set_linewidth(1.2)

plt.subplots_adjust(bottom=0.12)

# --- Save ---
plt.tight_layout()
output_filename = f"fqe_confidence_interval_comparison_{current_time}.pdf"
plt.savefig(output_filename, dpi=1200, bbox_inches='tight')
print(f"\nFQE Confidence Interval plot saved as '{output_filename}'")

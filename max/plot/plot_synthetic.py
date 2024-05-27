import matplotlib.pyplot as plt

# General settings for high-quality plots
plt.rcParams.update({
    'figure.figsize': (8, 4),             # Size of the figure
    'font.size': 12,                       # Font size for text
    'axes.labelsize': 16,                  # Font size for x and y labels
    'xtick.labelsize': 14,                 # Font size for x tick labels
    'ytick.labelsize': 14,                 # Font size for y tick labels
    'legend.fontsize': 12,                 # Font size for legend
    'lines.linewidth': 2,                  # Line width
    'lines.markersize': 6,                 # Marker size
    'axes.grid': False,                     # Add grid lines
    'grid.alpha': 0.7,                     # Transparency of grid lines
    'grid.linestyle': '--',                # Style of grid lines
    'grid.color': 'grey',                  # Color of grid lines
    'savefig.dpi': 300,                    # DPI for saved figures
    'savefig.bbox': 'tight',               # Tight bounding box for saved figures
})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

import numpy as np

# Example data
x = [1, 2, 3, 4, 5]
y = [0.261, 0.131, 0.0974, 0.0863, 0.0892]


# Create a plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Add titles and labels
ax.set_ylabel('10-step N-RMSE')
ax.set_xlabel('Initial Context Length')

# Add a legend
ax.set_xticks(x)
ax.set_ylim([0.0, 0.3])
# Save the figure
plt.savefig('high_quality_plot.pdf')
plt.tight_layout()
# Show the plot
plt.show()


#!/usr/bin/env python3
"""
Generate visualization plots for KV-Compress experimental results.
Creates comparison plots for Spline→PCA vs PCA and Flow→PCA vs PCA.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

def plot_compression_comparison():
    """Plot MSE comparison for both methods."""
    # Load data
    spline_df = pd.read_csv('results_spline_pca.csv')
    flow_df = pd.read_csv('results_affineflow_pca.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Spline→PCA vs PCA
    ax1.plot(spline_df['k'], spline_df['test_mse_PCA'],
             'o-', label='Standard PCA', color='#2E86AB', linewidth=2.5)
    ax1.plot(spline_df['k'], spline_df['test_mse_SplinePCA'],
             's-', label='Spline→PCA', color='#A23B72', linewidth=2.5)

    ax1.set_xlabel('Bottleneck Dimension (k)', fontweight='bold')
    ax1.set_ylabel('Test MSE (reconstruction error)', fontweight='bold')
    ax1.set_title('Spline→PCA vs Standard PCA', fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    # Annotate improvements at low k
    for i in [0, 1]:  # k=8, k=12
        k = spline_df['k'].iloc[i]
        pca_mse = spline_df['test_mse_PCA'].iloc[i]
        spline_mse = spline_df['test_mse_SplinePCA'].iloc[i]
        improvement = (pca_mse - spline_mse) / pca_mse * 100
        if improvement > 0.05:  # Only show if > 0.05%
            ax1.annotate(f'{improvement:.1f}% better',
                        xy=(k, spline_mse),
                        xytext=(k + 2, spline_mse * 0.95),
                        fontsize=9, color='#A23B72',
                        arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1))

    # Plot 2: Flow→PCA vs PCA
    ax2.plot(flow_df['k'], flow_df['test_mse_PCA'],
             'o-', label='Standard PCA', color='#2E86AB', linewidth=2.5)
    ax2.plot(flow_df['k'], flow_df['test_mse_FlowPCA'],
             '^-', label='Flow→PCA (RealNVP)', color='#F18F01', linewidth=2.5)

    ax2.set_xlabel('Bottleneck Dimension (k)', fontweight='bold')
    ax2.set_ylabel('Test MSE (reconstruction error)', fontweight='bold')
    ax2.set_title('Flow→PCA vs Standard PCA', fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('compression_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: compression_comparison.png")

def plot_delta_comparison():
    """Plot improvement delta (PCA - Method) to highlight wins."""
    spline_df = pd.read_csv('results_spline_pca.csv')
    flow_df = pd.read_csv('results_affineflow_pca.csv')

    # Calculate deltas (negative = improvement)
    spline_delta = spline_df['test_mse_SplinePCA'] - spline_df['test_mse_PCA']
    flow_delta = flow_df['test_mse_FlowPCA'] - flow_df['test_mse_PCA']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(spline_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, spline_delta * 1e6, width,
                   label='Spline→PCA', color='#A23B72', alpha=0.8)
    bars2 = ax.bar(x + width/2, flow_delta * 1e6, width,
                   label='Flow→PCA', color='#F18F01', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Bottleneck Dimension (k)', fontweight='bold')
    ax.set_ylabel('MSE Delta (×10⁻⁶)\n← Better | Worse →', fontweight='bold')
    ax.set_title('Improvement Over Standard PCA (negative = better)',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(spline_df['k'])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:  # Only label significant differences
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)

    plt.tight_layout()
    plt.savefig('improvement_delta.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: improvement_delta.png")

def plot_memory_reduction():
    """Plot memory reduction for ablation study C1-C3."""
    configs = ['V0\nBaseline', 'V19\nPrune', 'C1\nk=32', 'C2\nk=16', 'C3\nk=8']
    memory = [65536, 25024, 12512, 6256, 3128]  # Per head
    reduction = [0, 62, 81, 90, 95]  # Percentage

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute memory
    colors = ['#2E86AB', '#6A994E', '#BC4749', '#F18F01', '#A23B72']
    bars = ax1.bar(configs, memory, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('V Cache Size (per head)', fontweight='bold')
    ax1.set_title('KV-Geometry Ablation: Memory Usage', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels
    for bar, mem, red in zip(bars, memory, reduction):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:,}\n({red}% ↓)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Reduction percentage
    bars2 = ax2.bar(configs, reduction, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Memory Reduction (%)', fontweight='bold')
    ax2.set_title('Memory Reduction vs Baseline', fontweight='bold', pad=15)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.6, label='90% target')
    ax2.legend(loc='upper left')

    # Add value labels
    for bar, red in zip(bars2, reduction):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{red}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('memory_reduction.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: memory_reduction.png")

def plot_architecture_diagram():
    """Create a simple visual diagram of Spline→PCA pipeline."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Standard PCA pipeline (top)
    y_top = 3
    boxes = [
        (1, y_top, 'V\n(64-dim)', '#2E86AB'),
        (3.5, y_top, 'PCA', '#6A994E'),
        (6, y_top, 'V_c\n(k-dim)', '#BC4749'),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6,
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    # Arrows
    ax.arrow(1.5, y_top, 1.4, 0, head_width=0.15, head_length=0.2,
            fc='black', ec='black', linewidth=2)
    ax.arrow(4.1, y_top, 1.4, 0, head_width=0.15, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    ax.text(6.8, y_top, 'Standard PCA', fontsize=11, fontweight='bold',
           va='center', style='italic')

    # Spline→PCA pipeline (bottom)
    y_bot = 1
    boxes = [
        (1, y_bot, 'V\n(64-dim)', '#2E86AB'),
        (2.5, y_bot, 'Spline\nTransform', '#A23B72'),
        (4.5, y_bot, 'PCA', '#6A994E'),
        (6, y_bot, 'V_c\n(k-dim)', '#BC4749'),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6,
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    # Arrows
    for start, end in [(1.5, 2.0), (3.0, 3.9), (5.1, 5.5)]:
        ax.arrow(start, y_bot, end-start-0.1, 0, head_width=0.15, head_length=0.2,
                fc='black', ec='black', linewidth=2)

    ax.text(6.8, y_bot, 'Spline→PCA (better!)', fontsize=11, fontweight='bold',
           va='center', style='italic', color='#A23B72')

    # Title
    ax.text(6, 3.7, 'KV-Geometry Compression Pipeline',
           fontsize=14, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    print("✓ Saved: architecture_diagram.png")

if __name__ == '__main__':
    print("Generating KV-Compress visualization plots...")
    plot_compression_comparison()
    plot_delta_comparison()
    plot_memory_reduction()
    plot_architecture_diagram()
    print("\n✓ All plots generated successfully!")
    print("\nGenerated files:")
    print("  - compression_comparison.png (main results)")
    print("  - improvement_delta.png (delta bars)")
    print("  - memory_reduction.png (ablation memory)")
    print("  - architecture_diagram.png (pipeline visual)")

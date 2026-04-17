#!/usr/bin/env python3
"""
Generate line plot and bar chart comparing R² performance across drugs for different methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ============================================================================
# CONFIGURATION DICTIONARY
# ============================================================================

CONFIG = {
    # Method display names (CSV column name -> display name)
    'method_names': {
        'eNest': 'eNeST',
        'Uniform Random Sparse NN': 'Uniform SNN',
        'ERK SNN': 'ERK SNN',
        'Layer-wise Prune NN': 'Layer-wise Prune NN',
        'Global Prune NN': 'Global Prune NN',
        'Relaxed Global Prune NN': 'Relaxed Global Prune NN'
    },
    
    # Drug display names (CSV drug code -> display name)
    'drug_names': 
    {
        'D5': 'Topotecan',
        'D57': 'Veliparib',
        'D80': 'Bendamustine',
        'D99': 'CD-437',
        'D127': 'Clofarabine',
        'D151': 'Doxorubicin',
        'D188': 'Gemcitabine',
        'D201': 'Ifosfamide',
        'D244': 'Adavosertib',
        'D273': 'Mitomycin-C',
        'D298': 'Nelarabine',
        'D380': 'Camptothecin'
    },
    
    
    # Plot titles
    'line_plot_title': 'Dynamic v. Static Training Sparsity R² Performances',
    'bar_chart_title': 'Dynamic v. Static Training Sparsity R² Performances',
    
    # Axis labels
    'xlabel': 'Drug',
    'ylabel': 'R² Score',
    
    # Color scheme
    # eNest: positive color (star method) - using vibrant green
    # Uniform Random Sparse NN and ERK SNN: similar colors, closer to eNest - using blue-green shades
    # Other methods: clearly distinguishable colors
    'colors': {
        'eNest': '#2E7D32',  # Strong green (positive, star method)
        'Uniform Random Sparse NN': '#0288D1',  # Blue (similar to eNest, positive)
        'ERK SNN': '#00ACC1',  # Cyan-blue (similar to Uniform Random, positive)
        'Layer-wise Prune NN': '#D32F2F',  # Red (clearly different)
        'Global Prune NN': '#F57C00',  # Orange (clearly different)
        'Relaxed Global Prune NN': '#7B1FA2'  # Purple (clearly different)
    },
    
    # Method order for bar chart (left to right)
    'method_order': [
        'eNest',
        'Uniform Random Sparse NN',
        'ERK SNN',
        'Layer-wise Prune NN',
        'Global Prune NN',
        'Relaxed Global Prune NN'
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_r2_value(value_str):
    """Extract R² value from string like '0.4313 ± 0.0792 (p<0.001)'."""
    if pd.isna(value_str) or value_str == '':
        return np.nan
    # Extract the first number before ±
    match = re.match(r'([-+]?\d*\.?\d+)', str(value_str))
    if match:
        return float(match.group(1))
    return np.nan

def extract_std_dev(value_str):
    """Extract standard deviation from string like '0.4313 ± 0.0792 (p<0.001)'."""
    if pd.isna(value_str) or value_str == '':
        return np.nan
    # Extract the number after ±
    match = re.search(r'±\s*([-+]?\d*\.?\d+)', str(value_str))
    if match:
        return float(match.group(1))
    return np.nan

def load_and_parse_data(csv_file):
    """Load CSV and extract R² values and standard deviations for specified methods."""
    df = pd.read_csv(csv_file)
    
    # Get the methods we want to plot
    methods = list(CONFIG['method_names'].keys())
    
    # Extract R² values and standard deviations for each method
    data = {'Drug': df['Drug'].values}
    
    for method in methods:
        if method in df.columns:
            data[method] = df[method].apply(extract_r2_value).values
            data[f'{method}_std'] = df[method].apply(extract_std_dev).values
        else:
            print(f"Warning: Method '{method}' not found in CSV columns")
            data[method] = np.full(len(df), np.nan)
            data[f'{method}_std'] = np.full(len(df), np.nan)
    
    return pd.DataFrame(data)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_line_plot(df, output_file='r2_comparison_line_plot.png'):
    """Create line plot comparing methods across drugs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get drug order (sorted by drug code)
    drugs = sorted(df['Drug'].unique(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
    drug_labels = [CONFIG['drug_names'].get(d, d) for d in drugs]
    
    # Plot each method with error bars
    for method in CONFIG['method_order']:
        if method not in df.columns:
            continue
        
        display_name = CONFIG['method_names'].get(method, method)
        color = CONFIG['colors'].get(method, '#000000')
        std_col = f'{method}_std'
        
        # Get values and standard deviations for this method across all drugs
        values = []
        std_devs = []
        for drug in drugs:
            drug_data = df[df['Drug'] == drug]
            if len(drug_data) > 0:
                values.append(drug_data[method].iloc[0])
                if std_col in df.columns:
                    std_devs.append(drug_data[std_col].iloc[0])
                else:
                    std_devs.append(0)
            else:
                values.append(np.nan)
                std_devs.append(0)
        
        # Convert to numpy arrays for errorbar
        values = np.array(values)
        std_devs = np.array(std_devs)
        
        # Use errorbar to plot with error bars
        ax.errorbar(drug_labels, values, yerr=std_devs, marker='o', 
                   label=display_name, color=color, linewidth=2, 
                   markersize=8, capsize=4, capthick=1.5, elinewidth=1.5)
    
    ax.set_xlabel(CONFIG['xlabel'], fontsize=12)
    ax.set_ylabel(CONFIG['ylabel'], fontsize=12)
    ax.set_title(CONFIG['line_plot_title'], fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Line plot saved to: {output_file}")
    plt.close()

def create_bar_chart(df, output_file='r2_comparison_bar_chart.png'):
    """Create bar chart comparing methods for each drug."""
    # Get drug order (sorted by drug code)
    drugs = sorted(df['Drug'].unique(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
    drug_labels = [CONFIG['drug_names'].get(d, d) for d in drugs]
    
    # Set up the plot
    n_drugs = len(drugs)
    n_methods = len(CONFIG['method_order'])
    width = 0.8 / n_methods  # Width of each bar
    x = np.arange(n_drugs)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot bars for each method
    for i, method in enumerate(CONFIG['method_order']):
        if method not in df.columns:
            continue
        
        display_name = CONFIG['method_names'].get(method, method)
        color = CONFIG['colors'].get(method, '#000000')
        
        # Get values for this method across all drugs
        values = []
        for drug in drugs:
            drug_data = df[df['Drug'] == drug]
            if len(drug_data) > 0:
                values.append(drug_data[method].iloc[0])
            else:
                values.append(np.nan)
        
        # Calculate bar positions (offset for each method)
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=display_name, color=color, alpha=0.8)
    
    ax.set_xlabel(CONFIG['xlabel'], fontsize=12)
    ax.set_ylabel(CONFIG['ylabel'], fontsize=12)
    ax.set_title(CONFIG['bar_chart_title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drug_labels, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys
    
    # Default CSV file
    csv_file = 'r2_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"Loading data from: {csv_file}")
    df = load_and_parse_data(csv_file)
    
    print("\nGenerating plots...")
    create_line_plot(df)
    create_bar_chart(df)
    
    print("\nDone!")

if __name__ == '__main__':
    main()


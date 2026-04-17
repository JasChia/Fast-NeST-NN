#!/usr/bin/env python3
"""
Convert CSV tables to LaTeX pgfplotstableread format.
"""

import csv
import sys
import re

def format_value(cell_value, is_reference=False):
    """Convert CSV cell value to LaTeX format."""
    if not cell_value or cell_value.strip() == '':
        return ''
    
    # Replace ± with $\pm$
    cell_value = cell_value.replace('±', '$\\pm$')
    
    # Replace < with $<$ in p-values
    cell_value = cell_value.replace('p<', 'p$<$')
    
    # If this is the reference column, wrap in \textbf{} and remove p-value
    if is_reference:
        # Remove p-value part if present
        cell_value = re.sub(r'\s*\(p[^)]*\)', '', cell_value)
        return f'\\textbf{{{cell_value}}}'
    
    return cell_value

def wrap_column_name(name):
    """Wrap column names with spaces in braces."""
    if ' ' in name:
        return f'{{{name}}}'
    return name

def csv_to_latex(csv_file, reference_col='Old_eNest', output_name='datatableBHAdjusted'):
    """Convert CSV to LaTeX format."""
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        return ''
    
    # Get headers
    headers = rows[0]
    
    # Find reference column index
    try:
        ref_idx = headers.index(reference_col)
    except ValueError:
        ref_idx = None
    
    # Build LaTeX output
    output = ['\\pgfplotstableread[col sep=comma]{']
    
    # Header row
    header_line = ','.join([wrap_column_name(h) for h in headers])
    output.append(header_line)
    
    # Data rows
    for row in rows[1:]:
        if not row or not row[0]:  # Skip empty rows
            continue
        
        formatted_cells = []
        for i, cell in enumerate(row):
            is_ref = (i == ref_idx)
            formatted_cells.append(format_value(cell, is_ref))
        
        output.append(','.join(formatted_cells))
    
    output.append(f'}}\\{output_name}')
    
    return '\n'.join(output)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python csv_to_latex.py <input.csv> [reference_column] [output_name]")
        print("Example: python csv_to_latex.py r2_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv Old_eNest datatableBHAdjusted")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    reference_col = sys.argv[2] if len(sys.argv) > 2 else 'eNest'
    output_name = sys.argv[3] if len(sys.argv) > 3 else 'datatableBHAdjusted'
    
    latex_output = csv_to_latex(csv_file, reference_col, output_name)
    print(latex_output)


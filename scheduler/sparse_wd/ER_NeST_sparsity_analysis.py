"""
Analysis of NeST sparsity compared to Erdos-Renyi sparsity scaling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import binom

def calculate_sparsity_analysis(genotype_hiddens=4, input_dim=689):
	"""
	Calculate NeST sparsity for each layer and compare to Erdos-Renyi sparsity.
	
	Args:
		genotype_hiddens: Number of genotype hidden units
		input_dim: Input dimension (should be 689)
	"""
	
	# NeST edges per layer
	nest_edges_by_layer = {
		0: 1321 * genotype_hiddens,
		1: 92 * genotype_hiddens * genotype_hiddens,
		2: 36 * genotype_hiddens * genotype_hiddens,
		3: 13 * genotype_hiddens * genotype_hiddens,
		4: 4 * genotype_hiddens * genotype_hiddens,
		5: 2 * genotype_hiddens * genotype_hiddens,
		6: 2 * genotype_hiddens * genotype_hiddens,
		7: 1 * genotype_hiddens * genotype_hiddens,
	}
	
	# Nodes per layer
	nodes_per_layer = {
		0: input_dim,
		1: 76 * genotype_hiddens,
		2: 32 * genotype_hiddens,
		3: 13 * genotype_hiddens,
		4: 4 * genotype_hiddens,
		5: 2 * genotype_hiddens,
		6: 2 * genotype_hiddens,
		7: 1 * genotype_hiddens,
		8: 1 * genotype_hiddens,
	}
	
	# Fully connected edges per layer
	fc_edges_by_layer = {
		0: 689 * 76 * genotype_hiddens,
		1: 76 * genotype_hiddens * 32 * genotype_hiddens,
		2: 32 * genotype_hiddens * 13 * genotype_hiddens,
		3: 13 * genotype_hiddens * 4 * genotype_hiddens,
		4: 4 * genotype_hiddens * 2 * genotype_hiddens,
		5: 2 * genotype_hiddens * 2 * genotype_hiddens,
		6: 2 * genotype_hiddens * 1 * genotype_hiddens,
		7: 1 * genotype_hiddens * 1 * genotype_hiddens
	}
	
	print("=" * 80)
	print(f"NeST vs Erdos-Renyi Sparsity Analysis (genotype_hiddens={genotype_hiddens})")
	print("=" * 80)
	print()
	
	# Calculate NeST sparsity for each layer
	print("Layer-wise NeST Sparsity:")
	print("-" * 80)
	print(f"{'Layer':<8} {'FC Edges':<15} {'NeST Edges':<15} {'NeST Sparsity':<15} {'Nodes In':<12} {'Nodes Out':<12}")
	print("-" * 80)
	
	nest_sparsities = []
	layers = []
	
	for layer in range(8):
		fc_edges = fc_edges_by_layer[layer]
		nest_edges = nest_edges_by_layer[layer]
		nest_sparsity = 1.0 - (nest_edges / fc_edges) if fc_edges > 0 else 0.0
		
		nodes_in = nodes_per_layer[layer]
		nodes_out = nodes_per_layer[layer + 1]
		
		nest_sparsities.append(nest_sparsity)
		layers.append(layer)
		
		print(f"{layer:<8} {fc_edges:<15,} {nest_edges:<15,} {nest_sparsity:<15.4f} {nodes_in:<12} {nodes_out:<12}")
	
	print()
	
	# Calculate overall NeST sparsity (total edges across all layers)
	total_fc_edges = sum(fc_edges_by_layer.values())
	total_nest_edges = sum(nest_edges_by_layer.values())
	overall_nest_sparsity = 1.0 - (total_nest_edges / total_fc_edges) if total_fc_edges > 0 else 0.0
	
	# Calculate ER scaling using formula: 1 - (n_{l-1} + n_l) / (n_{l-1} * n_l)
	# Then scale by a constant to match target overall sparsity
	print("=" * 80)
	print("Erdos-Renyi Scaling Analysis (Formula-based, scaled to match overall sparsity):")
	print("=" * 80)
	print(f"Target Overall Sparsity: {overall_nest_sparsity:.4f} ({overall_nest_sparsity:.2%})")
	print()
	
	# First, calculate base formula edge probabilities for each layer
	base_er_edge_probs = {}
	base_er_sparsities = {}
	for layer in range(8):
		n_prev = nodes_per_layer[layer]
		n_curr = nodes_per_layer[layer + 1]
		
		# Compute base ER edge probability using formula: (n_{l-1} + n_l) / (n_{l-1} * n_l)
		if n_prev > 0 and n_curr > 0:
			base_edge_prob = (n_prev + n_curr) / (n_prev * n_curr)
			base_sparsity = 1.0 - base_edge_prob
		else:
			base_edge_prob = 1.0
			base_sparsity = 0.0
		
		base_er_edge_probs[layer] = base_edge_prob
		base_er_sparsities[layer] = base_sparsity
	
	# Calculate scaling constant using the suggested formula:
	# Sum of numerators: n_0 + 2*n_1 + 2*n_2 + ... + 2*n_{L-1} + n_L
	# (first and last layer nodes appear once, middle layers appear twice)
	sum_numerators = 0
	for layer in range(8):
		n_prev = nodes_per_layer[layer]
		n_curr = nodes_per_layer[layer + 1]
		if layer == 0:
			# First layer: only count n_0 once (n_0 + n_1, but n_1 will be counted in next layer)
			sum_numerators += n_prev
		if layer == 7:
			# Last layer: only count n_L once
			sum_numerators += n_curr
		else:
			# Middle layers: count n_curr (which becomes n_prev for next layer)
			sum_numerators += n_curr
	
	# Alternative: sum numerators directly from formula (n_{l-1} + n_l) for each layer
	# But accounting for double counting: n_0 appears once, n_1 through n_{L-1} appear twice, n_L appears once
	sum_numerators_alt = nodes_per_layer[0]  # n_0
	for layer in range(1, 8):
		sum_numerators_alt += 2 * nodes_per_layer[layer]  # Middle layers counted twice
	sum_numerators_alt += nodes_per_layer[8]  # n_L
	
	# Or simply: sum of (n_{l-1} + n_l) for all layers, but this double counts
	sum_numerators_direct = sum(nodes_per_layer[layer] + nodes_per_layer[layer + 1] for layer in range(8))
	
	# The correct sum (avoiding double counting) is: n_0 + 2*(n_1 + ... + n_{L-1}) + n_L
	sum_numerators_correct = nodes_per_layer[0] + nodes_per_layer[8]
	for i in range(1, 8):
		sum_numerators_correct += 2 * nodes_per_layer[i]
	
	# Scaling constant: target_sparsity * sum_fc_edges / sum_numerators
	# But we want edge probability scaling, so: (1 - target_sparsity) * sum_fc_edges / sum_numerators
	scaling_constant = (1.0 - overall_nest_sparsity) * total_fc_edges / sum_numerators_correct if sum_numerators_correct > 0 else 1.0
	
	print(f"Sum of numerators (n_0 + 2*(n_1+...+n_7) + n_8): {sum_numerators_correct:,}")
	print(f"Sum of FC edges: {total_fc_edges:,}")
	print(f"Target sparsity: {overall_nest_sparsity:.6f}")
	print(f"Scaling Constant: {scaling_constant:.6f}")
	print()
	
	# Now calculate scaled ER edges for each layer
	er_edges_by_layer_formula = {}
	er_sparsities_scaled = {}
	print(f"{'Layer':<8} {'n_{l-1}':<12} {'n_l':<12} {'Base ER Prob':<15} {'Scaled ER Prob':<15} {'Scaled ER Sparsity':<18} {'ER Edges':<15} {'FC Edges':<15} {'NeST Edges':<15}")
	print("-" * 120)
	
	for layer in range(8):
		n_prev = nodes_per_layer[layer]
		n_curr = nodes_per_layer[layer + 1]
		fc_edges = fc_edges_by_layer[layer]
		
		base_edge_prob = base_er_edge_probs[layer]
		scaled_edge_prob = min(1.0, scaling_constant * base_edge_prob)  # Cap at 1.0
		scaled_sparsity = 1.0 - scaled_edge_prob
		er_edges_scaled = scaled_edge_prob * fc_edges
		
		er_edges_by_layer_formula[layer] = int(er_edges_scaled)
		er_sparsities_scaled[layer] = scaled_sparsity
		nest_edges = nest_edges_by_layer[layer]
		
		# Warn if edge probability was capped
		capped_warning = " (CAPPED)" if scaling_constant * base_edge_prob > 1.0 else ""
		print(f"{layer:<8} {n_prev:<12} {n_curr:<12} {base_edge_prob:<15.6f} {scaled_edge_prob:<15.6f} {scaled_sparsity:<18.6f} {int(er_edges_scaled):<15,} {fc_edges:<15,} {nest_edges:<15,}{capped_warning}")
	
	print()
	
	# Verify overall sparsity matches target
	total_er_edges_scaled = sum(er_edges_by_layer_formula.values())
	overall_er_sparsity_scaled = 1.0 - (total_er_edges_scaled / total_fc_edges) if total_fc_edges > 0 else 0.0
	
	print(f"Total ER Edges (scaled formula): {total_er_edges_scaled:,}")
	print(f"Overall ER Sparsity (scaled formula): {overall_er_sparsity_scaled:.6f} ({overall_er_sparsity_scaled:.2%})")
	print(f"Target Overall Sparsity: {overall_nest_sparsity:.6f} ({overall_nest_sparsity:.2%})")
	print(f"Difference: {abs(overall_er_sparsity_scaled - overall_nest_sparsity):.6f}")
	print()
	
	# Print summary of scaled ER probabilities and sparsities
	print("=" * 80)
	print("Summary: Scaled ER Probabilities and Sparsities by Layer:")
	print("=" * 80)
	print(f"{'Layer':<8} {'Scaled ER Prob':<18} {'Scaled ER Sparsity':<18} {'ER Edges':<15} {'Note':<20}")
	print("-" * 100)
	for layer in range(8):
		scaled_prob = min(1.0, scaling_constant * base_er_edge_probs[layer])  # Use capped value
		scaled_sparsity = er_sparsities_scaled[layer]  # Use the stored capped sparsity
		er_edges = er_edges_by_layer_formula[layer]
		note = "Capped at 1.0" if scaling_constant * base_er_edge_probs[layer] > 1.0 else ""
		print(f"{layer:<8} {scaled_prob:<18.6f} {scaled_sparsity:<18.6f} {er_edges:<15,} {note:<20}")
	print()
	
	# Also calculate average NeST sparsity for comparison
	avg_nest_sparsity = np.mean(nest_sparsities)
	
	print("=" * 80)
	print("Overall NeST Sparsity Calculation:")
	print("=" * 80)
	print(f"Total FC Edges (across all layers): {total_fc_edges:,}")
	print(f"Total NeST Edges (across all layers): {total_nest_edges:,}")
	print(f"Overall NeST Sparsity: {overall_nest_sparsity:.4f}")
	print(f"Average NeST Sparsity (layer-wise): {avg_nest_sparsity:.4f}")
	print()
	
	# Compare to Erdos-Renyi sparsity levels
	print("=" * 80)
	print("Comparison with Erdos-Renyi Sparsity Levels:")
	print("=" * 80)
	print()
	
	# Common ER sparsity levels to compare
	er_sparsity_levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
	
	print(f"{'Layer':<8} {'NeST':<10} ", end="")
	for er_sp in er_sparsity_levels:
		print(f"ER {er_sp:.2f}  ", end="")
	print()
	print("-" * 80)
	
	for layer in range(8):
		fc_edges = fc_edges_by_layer[layer]
		nest_edges = nest_edges_by_layer[layer]
		nest_sparsity = 1.0 - (nest_edges / fc_edges) if fc_edges > 0 else 0.0
		
		print(f"{layer:<8} {nest_sparsity:<10.4f} ", end="")
		for er_sp in er_sparsity_levels:
			er_edges = int((1.0 - er_sp) * fc_edges)
			# Show how many edges ER would have
			print(f"{er_edges:<8,} ", end="")
		print()
	
	print()
	
	# Calculate what ER sparsity would give the same number of edges as NeST
	print("=" * 80)
	print("Equivalent Erdos-Renyi Sparsity (matching NeST edge counts):")
	print("=" * 80)
	print()
	print(f"{'Layer':<8} {'NeST Edges':<15} {'FC Edges':<15} {'Equivalent ER Sparsity':<20}")
	print("-" * 80)
	
	for layer in range(8):
		fc_edges = fc_edges_by_layer[layer]
		nest_edges = nest_edges_by_layer[layer]
		equivalent_er_sparsity = 1.0 - (nest_edges / fc_edges) if fc_edges > 0 else 0.0
		
		print(f"{layer:<8} {nest_edges:<15,} {fc_edges:<15,} {equivalent_er_sparsity:<20.4f}")
	
	print()
	
	# Analyze scaling behavior
	print("=" * 80)
	print("Scaling Analysis:")
	print("=" * 80)
	print()
	
	# Calculate how sparsity scales with number of nodes
	print("NeST sparsity vs layer size (nodes_in * nodes_out):")
	print(f"{'Layer':<8} {'Nodes In*Out':<15} {'NeST Sparsity':<15} {'Sparsity/Nodes':<15}")
	print("-" * 80)
	
	for layer in range(8):
		nodes_in = nodes_per_layer[layer]
		nodes_out = nodes_per_layer[layer + 1]
		total_nodes = nodes_in * nodes_out
		nest_sparsity = nest_sparsities[layer]
		sparsity_per_node = nest_sparsity / total_nodes if total_nodes > 0 else 0.0
		
		print(f"{layer:<8} {total_nodes:<15,} {nest_sparsity:<15.4f} {sparsity_per_node:<15.8f}")
	
	print()
	
	# Summary statistics
	print("=" * 80)
	print("Summary Statistics:")
	print("=" * 80)
	print(f"Min NeST Sparsity: {min(nest_sparsities):.4f} (Layer {layers[np.argmin(nest_sparsities)]})")
	print(f"Max NeST Sparsity: {max(nest_sparsities):.4f} (Layer {layers[np.argmax(nest_sparsities)]})")
	print(f"Mean NeST Sparsity: {np.mean(nest_sparsities):.4f}")
	print(f"Std NeST Sparsity: {np.std(nest_sparsities):.4f}")
	print()
	
	# Check if NeST maintains constant sparsity (like ER would)
	print("NeST sparsity variation:")
	if np.std(nest_sparsities) < 0.05:
		print("  → NeST maintains relatively constant sparsity (similar to constant ER)")
	else:
		print(f"  → NeST sparsity varies significantly (std={np.std(nest_sparsities):.4f})")
		print("  → This suggests NeST does NOT use constant Erdos-Renyi sparsity")
	
	return {
		'layers': layers,
		'nest_sparsities': nest_sparsities,
		'nest_edges_by_layer': nest_edges_by_layer,
		'fc_edges_by_layer': fc_edges_by_layer,
		'nodes_per_layer': nodes_per_layer,
		'overall_nest_sparsity': overall_nest_sparsity,
		'avg_nest_sparsity': avg_nest_sparsity,
		'total_fc_edges': total_fc_edges,
		'total_nest_edges': total_nest_edges,
		'er_edges_by_layer_formula': er_edges_by_layer_formula,
		'er_sparsities_scaled': er_sparsities_scaled,
		'scaling_constant': scaling_constant,
		'base_er_edge_probs': base_er_edge_probs
	}

def visualize_er_nest_comparison(genotype_hiddens=4, input_dim=689, results=None):
	"""
	Visualize PDF of number of connections by layer for Erdos-Renyi using scaled formula-based sparsity,
	and highlight NeST sparsity.
	
	Args:
		genotype_hiddens: Number of genotype hidden units
		input_dim: Input dimension (should be 689)
		results: Results dictionary from calculate_sparsity_analysis (if None, will compute)
	"""
	
	# If results not provided, compute them
	if results is None:
		results = calculate_sparsity_analysis(genotype_hiddens, input_dim)
	
	# NeST edges per layer
	nest_edges_by_layer = {
		0: 1321 * genotype_hiddens,
		1: 92 * genotype_hiddens * genotype_hiddens,
		2: 36 * genotype_hiddens * genotype_hiddens,
		3: 13 * genotype_hiddens * genotype_hiddens,
		4: 4 * genotype_hiddens * genotype_hiddens,
		5: 2 * genotype_hiddens * genotype_hiddens,
		6: 2 * genotype_hiddens * genotype_hiddens,
		7: 1 * genotype_hiddens * genotype_hiddens,
	}
	
	# Fully connected edges per layer
	fc_edges_by_layer = {
		0: 689 * 76 * genotype_hiddens,
		1: 76 * genotype_hiddens * 32 * genotype_hiddens,
		2: 32 * genotype_hiddens * 13 * genotype_hiddens,
		3: 13 * genotype_hiddens * 4 * genotype_hiddens,
		4: 4 * genotype_hiddens * 2 * genotype_hiddens,
		5: 2 * genotype_hiddens * 2 * genotype_hiddens,
		6: 2 * genotype_hiddens * 1 * genotype_hiddens,
		7: 1 * genotype_hiddens * 1 * genotype_hiddens
	}
	
	# Get scaled edge probabilities from results
	base_er_edge_probs = results['base_er_edge_probs']
	scaling_constant = results['scaling_constant']
	overall_nest_sparsity = results['overall_nest_sparsity']
	
	# Create figure with subplots for each layer
	fig, axes = plt.subplots(2, 4, figsize=(20, 10))
	fig.suptitle(f'NeST vs Erdos-Renyi (Formula-based, Scaled to {overall_nest_sparsity:.2%} Overall Sparsity)\nScaling Constant = {scaling_constant:.6f}', 
	             fontsize=16, fontweight='bold')
	
	axes = axes.flatten()
	
	for layer in range(8):
		ax = axes[layer]
		fc_edges = fc_edges_by_layer[layer]
		nest_edges = nest_edges_by_layer[layer]
		
		# Compute binomial distribution for Erdos-Renyi using scaled formula-based edge probability
		# Number of edges ~ Binomial(n=fc_edges, p=scaled_edge_prob)
		n = fc_edges
		scaled_edge_prob = min(1.0, scaling_constant * base_er_edge_probs[layer])  # Cap at 1.0
		p = scaled_edge_prob
		
		# Handle edge cases where p = 0.0 or p = 1.0 (deterministic)
		if p <= 0.0:
			# No edges: deterministic at 0
			mean = 0.0
			std = 0.0
			min_edges = 0
			max_edges = max(1, nest_edges)  # At least show NeST if it exists
		elif p >= 1.0:
			# All edges: deterministic at n
			mean = float(n)
			std = 0.0
			min_edges = max(0, n - 1)
			max_edges = n
		else:
			# Normal case: use mean ± 4 standard deviations
			mean = n * p
			std = np.sqrt(n * p * (1 - p))
			# Ensure NeST edges are in the range
			min_edges = max(0, min(int(mean - 4 * std), nest_edges - int(0.1 * n)))
			max_edges = min(n, max(int(mean + 4 * std), nest_edges + int(0.1 * n)))
		
		# For very large n, sample more sparsely
		# Handle deterministic cases (p=0 or p=1) where std=0
		if std == 0.0 or max_edges - min_edges <= 1:
			# Deterministic or very small range: use all values
			edge_range = np.arange(min_edges, max_edges + 1)
		elif n > 10000:
			# For large n, use normal approximation range but sample more densely
			step = max(1, int(std / 50))  # More samples for better visualization
			edge_range = np.arange(min_edges, max_edges + 1, step)
		elif n > 1000:
			step = max(1, int(std / 20))
			edge_range = np.arange(min_edges, max_edges + 1, step)
		else:
			edge_range = np.arange(min_edges, max_edges + 1)
		
		# Ensure nest_edges is in the range
		if nest_edges not in edge_range:
			edge_range = np.sort(np.unique(np.concatenate([edge_range, [nest_edges]])))
		
		# Compute PMF (handle deterministic cases)
		if p <= 0.0:
			# Deterministic: all probabilities are 0 except at 0
			pmf = np.where(edge_range == 0, 1.0, 0.0)
		elif p >= 1.0:
			# Deterministic: all probabilities are 0 except at n
			pmf = np.where(edge_range == n, 1.0, 0.0)
		else:
			# Normal binomial distribution
			pmf = binom.pmf(edge_range, n, p)
		
		# Calculate ER sparsity for this layer
		er_sparsity_layer = 1.0 - scaled_edge_prob
		
		# Plot Erdos-Renyi PDF
		ax.plot(edge_range, pmf, 'b-', linewidth=2, label=f'ER (sparsity={er_sparsity_layer:.4f})', alpha=0.7)
		ax.fill_between(edge_range, 0, pmf, alpha=0.3, color='blue')
		
		# Highlight NeST edge count
		# Find closest point in edge_range to nest_edges
		idx = np.argmin(np.abs(edge_range - nest_edges))
		# Calculate exact PMF value at NeST edges (handle deterministic cases)
		if p <= 0.0:
			nest_pmf_value = 1.0 if nest_edges == 0 else 0.0
		elif p >= 1.0:
			nest_pmf_value = 1.0 if nest_edges == n else 0.0
		else:
			nest_pmf_value = binom.pmf(nest_edges, n, p)
		
		ax.axvline(nest_edges, color='red', linestyle='--', linewidth=2.5, 
		          label=f'NeST: {nest_edges:,} edges', zorder=5)
		ax.plot(nest_edges, nest_pmf_value, 'ro', markersize=10, zorder=6, 
		       markeredgecolor='darkred', markeredgewidth=2)
		
		# Calculate NeST sparsity for this layer
		nest_sparsity = 1.0 - (nest_edges / fc_edges) if fc_edges > 0 else 0.0
		
		ax.set_xlabel('Number of Edges', fontsize=10)
		ax.set_ylabel('Probability Density', fontsize=10)
		ax.set_title(f'Layer {layer}\nFC: {fc_edges:,} | NeST: {nest_edges:,} | NeST Sparsity: {nest_sparsity:.4f}', 
		            fontsize=11, fontweight='bold')
		ax.legend(fontsize=9, loc='best')
		ax.grid(True, alpha=0.3)
		
		# Add statistics text
		er_mean = n * p
		if p <= 0.0 or p >= 1.0:
			er_std = 0.0  # Deterministic case
		else:
			er_std = np.sqrt(n * p * (1 - p))
		stats_text = f'ER Mean: {er_mean:.0f}\nER Std: {er_std:.0f}\nER Sparsity: {er_sparsity_layer:.4f}'
		ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
		       fontsize=8, verticalalignment='top', 
		       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
	
	plt.tight_layout()
	
	# Save figure
	output_file = f'ER_NeST_sparsity_comparison_formula_scaled_{overall_nest_sparsity:.4f}.png'
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	print(f"\nVisualization saved to: {output_file}")
	
	plt.close(fig)  # Close figure to free memory
	
	return fig

if __name__ == "__main__":
	# Run analysis for default genotype_hiddens=4
	results = calculate_sparsity_analysis(genotype_hiddens=4, input_dim=689)
	
	# Use overall NeST sparsity for ER comparison
	overall_nest_sparsity = results['overall_nest_sparsity']
	
	print("\n" + "=" * 80)
	print("Generating visualization...")
	print("=" * 80)
	print(f"Overall NeST Sparsity: {overall_nest_sparsity:.4f} ({overall_nest_sparsity:.2%})")
	print(f"Using formula-based ER scaled to match {overall_nest_sparsity:.2%} overall sparsity")
	print(f"Scaling constant: {results['scaling_constant']:.6f}")
	print("=" * 80)
	
	# Create visualization with formula-based ER scaled to match overall NeST sparsity
	visualize_er_nest_comparison(genotype_hiddens=4, input_dim=689, results=results)
	
	print("\n" + "=" * 80)
	print("Analysis complete!")
	print("=" * 80)


"""
NeST Sparsity
Layer 0: 0.9748
Layer 1: 0.9622
Layer 2: 0.9135
Layer 3: 0.7500
Layer 4: 0.5000
Layer 5: 0.5000
Layer 6: 0.0000
Layer 7: 0.0000
"""
#ER Sum: 1.437117
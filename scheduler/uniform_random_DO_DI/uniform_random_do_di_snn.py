"""Uniform Random Direct Output Direct Input Sparse Neural Network implementation.

This module implements a sparse neural network using uniform random pruning for initialization
of sparse network topology. The network includes:
1. Direct Output connections: nodes in each intermediary layer are split into non-overlapping 
   sets of size genotype_hiddens, and each set attempts to predict the final output through 
   an additional linear layer + activation.
2. Direct Input connections: each layer receives the original input concatenated with the 
   previous layer's output, with specified sparse connections from the input to each layer.

Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class UniformRandomDODISNN(nn.Module):
	"""Sparse Neural Network with Uniform Random Pruning, Direct Output, and Direct Input Connections.

	This network uses uniform random pruning to create sparse network topologies.
	
	Direct Input: Each layer receives the original input (genes) concatenated with the 
	previous layer's output. The input dimension for layer i is d + nodes_per_layer[i-1],
	where d is the number of genes (689). The connectivity is specified separately:
	- gene_to_layer_edges[i]: number of connections from genes to layer i
	- layer_to_layer_edges[i-1]: number of connections from layer i-1 to layer i
	
	Direct Output: Nodes in each intermediary layer are split into non-overlapping sets 
	of size genotype_hiddens. Each set produces a prediction which are all aggregated
	at the end through a linear layer + sigmoid.

	Args:
		input_dim: Input dimension (must be 689)
		dropout_fraction: Dropout rate for regularization
		activation: Activation function class (e.g., nn.Tanh, nn.ReLU)
		genotype_hiddens: Number of nodes per assembly (default: 4)
		seed: Random seed for reproducibility
		max_attempts: Maximum attempts to generate valid network topology
	"""

	def __init__(
		self,
		input_dim: int = 689,
		dropout_fraction: float = 0.0,
		activation: nn.Module = nn.Tanh,
		genotype_hiddens: int = 4,
		seed: Optional[int] = None,
		max_attempts: int = 10000,
	):
		super().__init__()

		assert input_dim == 689, "input_dim must be 689"
		assert isinstance(genotype_hiddens, int), "genotype_hiddens must be an integer"
		assert genotype_hiddens > 0, "genotype_hiddens must be positive"

		self.input_dim = input_dim
		self.activation = activation
		self.genotype_hiddens = genotype_hiddens
		self.max_attempts = max_attempts

		# Define Gene → Layer edges (Direct Input connections)
		# These are the number of connections from the input (genes) directly to each layer
		self.gene_to_layer_edges = {
			1: 637 * self.genotype_hiddens,   # Gene → Layer 1
			2: 253 * self.genotype_hiddens,   # Gene → Layer 2
			3: 112 * self.genotype_hiddens,   # Gene → Layer 3
			4: 92 * self.genotype_hiddens,    # Gene → Layer 4
			5: 39 * self.genotype_hiddens,    # Gene → Layer 5
			6: 35 * self.genotype_hiddens,    # Gene → Layer 6
			7: 44 * self.genotype_hiddens,    # Gene → Layer 7
			8: 109 * self.genotype_hiddens,   # Gene → Layer 8
		}

		# Define Layer → Layer edges (connections between consecutive layers)
		# layer_to_layer_edges[i] = connections from Layer i to Layer i+1
		self.layer_to_layer_edges = {
			1: 92 * self.genotype_hiddens * self.genotype_hiddens,   # Layer 1 → Layer 2
			2: 36 * self.genotype_hiddens * self.genotype_hiddens,   # Layer 2 → Layer 3
			3: 13 * self.genotype_hiddens * self.genotype_hiddens,   # Layer 3 → Layer 4
			4: 4 * self.genotype_hiddens * self.genotype_hiddens,    # Layer 4 → Layer 5
			5: 2 * self.genotype_hiddens * self.genotype_hiddens,    # Layer 5 → Layer 6
			6: 2 * self.genotype_hiddens * self.genotype_hiddens,    # Layer 6 → Layer 7
			7: 1 * self.genotype_hiddens * self.genotype_hiddens,    # Layer 7 → Layer 8
		}

		# Define network architecture: nodes per layer
		self.nodes_per_layer = {
			0: input_dim,  # Input layer (genes), should be 689!
			1: 76 * self.genotype_hiddens,
			2: 32 * self.genotype_hiddens,
			3: 13 * self.genotype_hiddens,
			4: 4 * self.genotype_hiddens,
			5: 2 * self.genotype_hiddens,
			6: 2 * self.genotype_hiddens,
			7: 1 * self.genotype_hiddens,
			8: 1 * self.genotype_hiddens,
		}

		# Validate gene → layer edges don't exceed fully-connected capacity
		for layer_idx in self.gene_to_layer_edges:
			max_edges = self.input_dim * self.nodes_per_layer[layer_idx]
			assert self.gene_to_layer_edges[layer_idx] <= max_edges, \
				f"gene_to_layer_edges[{layer_idx}] ({self.gene_to_layer_edges[layer_idx]}) > max ({max_edges})"

		# Validate layer → layer edges don't exceed fully-connected capacity
		for layer_idx in self.layer_to_layer_edges:
			max_edges = self.nodes_per_layer[layer_idx] * self.nodes_per_layer[layer_idx + 1]
			assert self.layer_to_layer_edges[layer_idx] <= max_edges, \
				f"layer_to_layer_edges[{layer_idx}] ({self.layer_to_layer_edges[layer_idx]}) > max ({max_edges})"

		# Build the sparse network masks using uniform random pruning
		self.combined_masks, self.num_attempts = self.build_sparse_network_masks(seed=seed)
		
		# Build a static sparse network (no prune/regrow during training)
		# Masks are applied once at init using PyTorch pruning; they stay fixed.
		self.sparse_layers, self.activation_layers, self.batchnorm_layers, self.dropout_layers = \
			self.build_neural_network(self.combined_masks, dropout_fraction=dropout_fraction)
		
		# Final output layer
		final_layer_nodes = self.nodes_per_layer[len(self.nodes_per_layer) - 1]
		self.final_linear = nn.Linear(final_layer_nodes, 1)
		self.final_activation = nn.Sigmoid()
		
		# Build direct output prediction layers (pre-compute for efficiency)
		# These layers split nodes into groups of genotype_hiddens and predict final output
		self.direct_output_layers, self.group_masks, self.total_groups = self.build_direct_output_layers()
		
		# Build aggregation layer that combines all predictions (intermediary + main)
		# Input: total_groups (from intermediary outputs) + 1 (from main output)
		# Output: 1 (final prediction)
		aggregation_input_dim = self.total_groups + 1 if self.total_groups > 0 else 1
		self.aggregation_linear = nn.Linear(aggregation_input_dim, 1)
		self.aggregation_activation = nn.Sigmoid()

	def build_sparse_network_masks(
		self,
		seed: Optional[int] = None,
	) -> Tuple[List[torch.Tensor], int]:
		"""Build sparse network masks using uniform random pruning.

		For each layer i, creates a combined mask with shape (input_dim_i, output_nodes):
		- Layer 1: input_dim_1 = d (genes only), mask shape (689, nodes_layer_1)
		- Layer i (i > 1): input_dim_i = d + nodes_layer_{i-1}, 
		  mask shape (689 + nodes_layer_{i-1}, nodes_layer_i)
		  
		The mask is constructed by:
		1. Generating gene_to_layer_edges[i] random connections from genes to layer i
		2. For layers 2+, generating layer_to_layer_edges[i-1] connections from layer i-1 to layer i
		3. Combining these into a single mask
		
		The aliveness check only validates the layer-to-layer part (ensuring each node
		has at least one input from the previous layer).

		Args:
			seed: Random seed for reproducibility. If None, uses random seed.

		Returns:
			Tuple containing:
				- combined_masks: List of weight masks for each layer
				- num_attempts: Number of resampling attempts needed to find valid network

		Raises:
			RuntimeError: If valid network cannot be generated after max_attempts.
		"""
		# Set random seed for reproducibility
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)
		
		num_attempts = 0

		for attempt in range(self.max_attempts):
			if attempt % 1000 == 0 and attempt > 0:
				print(f"  Attempt {attempt}/{self.max_attempts} - still searching for valid network...")

			num_attempts = attempt + 1
			combined_masks: List[torch.Tensor] = []
			valid_network = True

			# Generate masks for each hidden layer (layers 1-8)
			for layer_idx in range(1, len(self.nodes_per_layer)):
				output_nodes = self.nodes_per_layer[layer_idx]
				
				# Generate gene → layer mask
				gene_total_edges = self.input_dim * output_nodes
				gene_desired_edges = self.gene_to_layer_edges[layer_idx]
				
				temp_gene_linear = nn.Linear(self.input_dim, output_nodes)
				amount_to_prune = gene_total_edges - gene_desired_edges
				prune.random_unstructured(temp_gene_linear, name="weight", amount=amount_to_prune)
				gene_mask = getattr(temp_gene_linear, 'weight_mask').T  # shape: (input_dim, output_nodes)
				
				if layer_idx == 1:
					# Layer 1: only gene connections
					combined_mask = gene_mask
				else:
					# Layers 2+: gene + prev layer connections
					prev_layer_nodes = self.nodes_per_layer[layer_idx - 1]
					
					# Generate layer → layer mask
					layer_total_edges = prev_layer_nodes * output_nodes
					layer_desired_edges = self.layer_to_layer_edges[layer_idx - 1]
					
					temp_layer_linear = nn.Linear(prev_layer_nodes, output_nodes)
					amount_to_prune = layer_total_edges - layer_desired_edges
					prune.random_unstructured(temp_layer_linear, name="weight", amount=amount_to_prune)
					layer_mask = getattr(temp_layer_linear, 'weight_mask').T  # shape: (prev_layer_nodes, output_nodes)
					
					# Check aliveness: each output node must have at least one connection from prev layer
					has_prev_layer_connection = torch.any(layer_mask, dim=0)  # shape: (output_nodes,)
					if not torch.all(has_prev_layer_connection):
						valid_network = False
						break
					
					# Combine masks: [gene_mask; layer_mask] 
					# Shape: (input_dim + prev_layer_nodes, output_nodes)
					combined_mask = torch.cat([gene_mask, layer_mask], dim=0)
				
				combined_masks.append(combined_mask)
			
			if valid_network:
				break

		if not valid_network:
			print(f"❌ Failed to generate valid network after {self.max_attempts} attempts")
			raise RuntimeError(f"Failed to generate valid network after {self.max_attempts} attempts")
		else:
			print(f"✓ Successfully generated valid network after {num_attempts} attempts")

		return combined_masks, num_attempts

	def build_neural_network(
		self, 
		combined_masks: List[torch.Tensor],
		dropout_fraction: float = 0.0
	) -> Tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList, nn.ModuleList]:
		"""Build the neural network using combined connectivity masks.

		Each layer is a standard pruned linear layer. The input to each layer is:
		- Layer 1: genes only (shape: batch, 689)
		- Layer i (i > 1): concat(genes, prev_layer_output) (shape: batch, 689 + nodes_prev)

		Args:
			combined_masks: List of weight masks for each layer
			dropout_fraction: Dropout rate for regularization

		Returns:
			Tuple of ModuleLists containing sparse layers, activations, batchnorms, and dropouts
		"""
		sparse_layers = nn.ModuleList()
		activation_layers = nn.ModuleList()
		batchnorm_layers = nn.ModuleList()
		dropout_layers = nn.ModuleList()

		# Build sparse layers for each hidden layer (layers 1-8)
		for layer_idx in range(1, len(self.nodes_per_layer)):
			output_nodes = self.nodes_per_layer[layer_idx]
			
			# Determine input dimension
			if layer_idx == 1:
				input_nodes = self.input_dim  # 689
			else:
				input_nodes = self.input_dim + self.nodes_per_layer[layer_idx - 1]
			
			mask = combined_masks[layer_idx - 1]  # 0-indexed in list
			
			# Create linear layer and apply mask
			linear = nn.Linear(input_nodes, output_nodes)
			prune.custom_from_mask(linear, name="weight", mask=mask.T)
			
			sparse_layers.append(linear)
			activation_layers.append(self.activation())
			batchnorm_layers.append(nn.BatchNorm1d(output_nodes))
			dropout_layers.append(nn.Dropout(p=dropout_fraction))

		return sparse_layers, activation_layers, batchnorm_layers, dropout_layers

	def build_direct_output_layers(self) -> Tuple[nn.Module, List[torch.Tensor], int]:
		"""Build direct output prediction layers for all intermediary layers in parallel.

		Pre-computes masks and builds a single linear layer to process all groups
		from all intermediary layers in parallel, similar to eNest's asm_out_linear_layer approach.

		Returns:
			Tuple containing:
				- direct_output_layers: Sequential module with Linear + Activation
				- group_masks: List of masks for extracting groups from each layer's activations
				- total_groups: Total number of groups across all intermediary layers
		"""
		group_masks = []
		total_groups = 0
		
		# Process each intermediary layer (skip input layer 0 and final layer)
		# Intermediary layers are indices 1 through len(nodes_per_layer) - 2
		for layer_idx in range(1, len(self.nodes_per_layer) - 1):
			num_nodes = self.nodes_per_layer[layer_idx]
			num_groups = num_nodes // self.genotype_hiddens
			
			# Only create masks if we have at least one complete group
			if num_groups > 0:
				# Create mask for this layer: (num_groups, num_nodes)
				# Each row selects one group of genotype_hiddens nodes
				layer_mask = torch.zeros(num_groups, num_nodes)
				for group_idx in range(num_groups):
					start_idx = group_idx * self.genotype_hiddens
					end_idx = start_idx + self.genotype_hiddens
					layer_mask[group_idx, start_idx:end_idx] = 1.0
				
				group_masks.append(layer_mask)
				total_groups += num_groups
			else:
				# Empty mask for layers with no complete groups
				group_masks.append(None)
		
		# Build a single linear layer to process all groups in parallel
		# Input: total_groups * genotype_hiddens (flattened groups)
		# Output: total_groups (one prediction per group)
		# Then apply activation
		if total_groups > 0:
			direct_output_linear = nn.Linear(total_groups * self.genotype_hiddens, total_groups)
			direct_output_layers = nn.Sequential(
				direct_output_linear,
				self.activation()
			)
		else:
			# Dummy layer if no groups
			direct_output_layers = nn.Sequential(
				nn.Linear(1, 1),
				self.activation()
			)
		
		return direct_output_layers, group_masks, total_groups

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""Forward pass through the neural network.

		Each hidden layer receives the concatenation of the original input (genes) and 
		the previous layer's output. Collects outputs from all layers, transforms them 
		through direct output layers (non-overlapping sets of genotype_hiddens nodes),
		and aggregates all predictions through a final linear layer + sigmoid.

		Args:
			X: Input tensor of shape (batch_size, input_dim)

		Returns:
			Final aggregated output tensor of shape (batch_size, 1)
			This is the result of aggregating all layer predictions through
			a linear layer and sigmoid activation.
		"""
		# Store original input for direct input connections
		gene_input = X
		
		# Forward pass through main network, collecting activations from all layers
		all_layer_activations = []
		prev_layer_output = None
		
		# Process each hidden layer
		for layer_idx in range(len(self.sparse_layers)):
			# Build layer input: genes only for layer 1, concat(genes, prev_output) for others
			if layer_idx == 0:
				layer_input = gene_input
			else:
				layer_input = torch.cat([gene_input, prev_layer_output], dim=1)
			
			# Forward through sparse linear layer
			current_output = self.sparse_layers[layer_idx](layer_input)
			
			# Activation
			current_output = self.activation_layers[layer_idx](current_output)
			
			# BatchNorm
			current_output = self.batchnorm_layers[layer_idx](current_output)
			
			# Dropout
			current_output = self.dropout_layers[layer_idx](current_output)
			
			# Collect activation from all layers (including the last one)
			all_layer_activations.append(current_output)
			
			# Update previous layer output for next iteration
			prev_layer_output = current_output
		
		# Process all layer activations to get predictions
		all_predictions = []
		
		# Process intermediary layers (all except the last one) through direct output layers
		if self.total_groups > 0:
			# Collect all groups from intermediary layers
			all_groups_flat = []
			for layer_idx in range(len(all_layer_activations) - 1):  # Exclude final layer
				layer_activation = all_layer_activations[layer_idx]
				layer_mask = self.group_masks[layer_idx]
				if layer_mask is not None:
					batch_size = layer_activation.shape[0]
					num_groups = layer_mask.shape[0]
					
					# Extract groups: (batch_size, num_nodes) -> (batch_size, num_groups * genotype_hiddens)
					num_nodes_to_use = num_groups * self.genotype_hiddens
					groups_reshaped = layer_activation[:, :num_nodes_to_use].view(
						batch_size, num_groups, self.genotype_hiddens
					)
					groups_flat = groups_reshaped.view(batch_size, -1)
					all_groups_flat.append(groups_flat)
			
			# Process all intermediary layer groups through direct output layers
			if all_groups_flat:
				all_groups_concat = torch.cat(all_groups_flat, dim=1)  # (batch_size, total_groups * genotype_hiddens)
				intermediary_predictions = self.direct_output_layers(all_groups_concat)  # (batch_size, total_groups)
				all_predictions.append(intermediary_predictions)
		
		# Process final layer through main output layer
		final_layer_activation = all_layer_activations[-1]
		final_output_pre_sigmoid = self.final_linear(final_layer_activation)  # (batch_size, 1)
		all_predictions.append(final_output_pre_sigmoid)
		
		# Concatenate all predictions: (batch_size, total_groups + 1)
		all_predictions_concat = torch.cat(all_predictions, dim=1)
		
		# Pass through aggregation layer and apply sigmoid
		main_output = self.aggregation_activation(self.aggregation_linear(all_predictions_concat))
		
		return main_output

	def finalize_pruning(self) -> None:
		"""Remove pruning re-parametrization and bake masked weights in place.

		Call this before saving the final/eval model to strip mask buffers.
		This permanently removes the pruning masks and keeps only the active weights.
		"""
		for sparse_layer in self.sparse_layers:
			if hasattr(sparse_layer, "weight_mask"):
				prune.remove(sparse_layer, "weight")


# ---------------------- Testing ---------------------- #
if __name__ == "__main__":
	print("=== Testing Uniform Random DO+DI (Direct Output + Direct Input) Sparse NN ===")
	seed = 42
	genotype_hiddens = 4

	# Test 1: Multiple instances with same seed (reproducibility)
	print("\n=== Test 1: Reproducibility ===")
	results = []
	for i in range(3):
		sparse_nn = UniformRandomDODISNN(genotype_hiddens=genotype_hiddens, seed=seed)
		mask_sums = [mask.sum().item() for mask in sparse_nn.combined_masks]
		results.append((sparse_nn.num_attempts, mask_sums))
		print(f"Instance {i+1}: {sparse_nn.num_attempts} attempts, mask sums: {mask_sums[:3]}...")

	attempts_same = all(r[0] == results[0][0] for r in results)
	masks_same = all(r[1] == results[0][1] for r in results)
	print(f"Same attempts: {attempts_same}")
	print(f"Same mask sums: {masks_same}")
	print(f"✅ Reproducible: {attempts_same and masks_same}")

	# Test 2: Different seeds produce different results
	print("\n=== Test 2: Different Seeds ===")
	seeds = [42, 123, 456]
	for seed in seeds:
		sparse_nn = UniformRandomDODISNN(genotype_hiddens=genotype_hiddens, seed=seed)
		print(f"Seed {seed}: {sparse_nn.num_attempts} attempts")

	# Test 3: Test forward pass
	print("\n=== Test 3: Forward Pass ===")
	sparse_nn_test = UniformRandomDODISNN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, sparse_nn_test.input_dim)
	try:
		main_output = sparse_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Main output shape: {main_output.shape}")
		print(f"Main output range: [{main_output.min().item():.4f}, {main_output.max().item():.4f}]")
		print(f"✅ Network is ready to use!")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
		import traceback
		traceback.print_exc()

	# Test 4: Verify connectivity structure
	print("\n=== Test 4: Connectivity Verification ===")
	sparse_nn_verify = UniformRandomDODISNN(genotype_hiddens=genotype_hiddens, seed=42)
	print("Layer input dimensions and mask shapes:")
	for i, mask in enumerate(sparse_nn_verify.combined_masks):
		layer_idx = i + 1
		expected_input = sparse_nn_verify.input_dim if layer_idx == 1 else \
			sparse_nn_verify.input_dim + sparse_nn_verify.nodes_per_layer[layer_idx - 1]
		expected_output = sparse_nn_verify.nodes_per_layer[layer_idx]
		actual_shape = mask.shape
		print(f"  Layer {layer_idx}: expected ({expected_input}, {expected_output}), actual {actual_shape}, match: {actual_shape == (expected_input, expected_output)}")
	
	print("\nEdge counts verification:")
	for i, mask in enumerate(sparse_nn_verify.combined_masks):
		layer_idx = i + 1
		expected_gene_edges = sparse_nn_verify.gene_to_layer_edges[layer_idx]
		
		if layer_idx == 1:
			actual_total = int(mask.sum().item())
			print(f"  Layer {layer_idx}: gene edges = {expected_gene_edges}, actual total = {actual_total}, match: {expected_gene_edges == actual_total}")
		else:
			# Split mask into gene part and layer part
			gene_part = mask[:sparse_nn_verify.input_dim, :]
			layer_part = mask[sparse_nn_verify.input_dim:, :]
			
			actual_gene_edges = int(gene_part.sum().item())
			actual_layer_edges = int(layer_part.sum().item())
			expected_layer_edges = sparse_nn_verify.layer_to_layer_edges[layer_idx - 1]
			
			print(f"  Layer {layer_idx}: gene edges = {expected_gene_edges} (actual: {actual_gene_edges}), "
				  f"layer edges = {expected_layer_edges} (actual: {actual_layer_edges}), "
				  f"match: {expected_gene_edges == actual_gene_edges and expected_layer_edges == actual_layer_edges}")

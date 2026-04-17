"""Uniform Random Direct Output Sparse Neural Network implementation.

This module implements a sparse neural network using uniform random pruning for initialization
of sparse network topology. The network includes direct output connections where nodes in each
intermediary layer are split into non-overlapping sets of size genotype_hiddens, and each set
attempts to predict the final output through an additional linear layer + activation.
Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
# This code uses uniform random pruning for sparse network topology with direct output connections
class UniformRandomSNN(nn.Module):
	"""Sparse Neural Network with Uniform Random Pruning and Direct Output Connections.

	This network uses uniform random pruning to create sparse network topologies.
	The network includes direct output connections where nodes in each intermediary layer
	are split into non-overlapping sets of size genotype_hiddens. Each set attempts
	to predict the final output through an additional linear layer + activation,
	which can be used for auxiliary losses during training.

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

		self.activation = activation
		self.genotype_hiddens = genotype_hiddens
		self.max_attempts = max_attempts

		# Define network architecture: edges per layer
		self.nest_edges_by_layer = {
			0: 1321 * self.genotype_hiddens,  # 7926,5284
			1: 92 * self.genotype_hiddens * self.genotype_hiddens,  # 3312, 1472
			2: 36 * self.genotype_hiddens * self.genotype_hiddens,  # 1296, 576
			3: 13 * self.genotype_hiddens * self.genotype_hiddens,  # 468, 208
			4: 4 * self.genotype_hiddens * self.genotype_hiddens,  # 144, 64
			5: 2 * self.genotype_hiddens * self.genotype_hiddens,  # 72, 32
			6: 2 * self.genotype_hiddens * self.genotype_hiddens,  # Full, 72, 32
			7: 1 * self.genotype_hiddens * self.genotype_hiddens,  # Full, 36, 16
		}

		# Define network architecture: nodes per layer
		self.nodes_per_layer = {
			0: input_dim,  # Input layer, should be 689!
			1: 76 * self.genotype_hiddens,
			2: 32 * self.genotype_hiddens,
			3: 13 * self.genotype_hiddens,
			4: 4 * self.genotype_hiddens,
			5: 2 * self.genotype_hiddens,
			6: 2 * self.genotype_hiddens,
			7: 1 * self.genotype_hiddens,
			8: 1 * self.genotype_hiddens,
		}

		# Define fully-connected edges per layer (for validation)
		self.fc_edges_by_layer = {
			0: 689 * 76 * self.genotype_hiddens,
			1: 76 * self.genotype_hiddens * 32 * self.genotype_hiddens,
			2: 32 * self.genotype_hiddens * 13 * self.genotype_hiddens,
			3: 13 * self.genotype_hiddens * 4 * self.genotype_hiddens,
			4: 4 * self.genotype_hiddens * 2 * self.genotype_hiddens,
			5: 2 * self.genotype_hiddens * 2 * self.genotype_hiddens,
			6: 2 * self.genotype_hiddens * 1 * self.genotype_hiddens,
			7: 1 * self.genotype_hiddens * 1 * self.genotype_hiddens,
		}

		# Validate that sparse edges don't exceed fully-connected edges
		for k in self.nest_edges_by_layer:
			assert (
				self.nest_edges_by_layer[k] <= self.fc_edges_by_layer[k]
			), f"nest_edges_by_layer[{k}] > fc_edges_by_layer[{k}]"

		# Build the sparse network masks using uniform random pruning
		self.connectivity_list, self.num_attempts = self.build_sparse_network_masks(seed=seed)
		# Build a static sparse network (no prune/regrow during training)
		# Masks are applied once at init using PyTorch pruning; they stay fixed.
		self.NN = self.build_neural_network(self.connectivity_list, dropout_fraction=dropout_fraction)
		
		# Build direct output prediction layers (pre-compute for efficiency)
		# These layers split nodes into groups of genotype_hiddens and predict final output
		self.direct_output_layers, self.group_masks, self.total_groups = self.build_direct_output_layers()

	def build_sparse_network_masks(
		self,
		nodes_per_layer: Optional[Dict[int, int]] = None,
		nest_edges_by_layer: Optional[Dict[int, int]] = None,
		seed: Optional[int] = None,
	) -> Tuple[List[torch.Tensor], int]:
		"""Build sparse network masks using uniform random pruning.

		This method uses uniform random pruning to create sparse network topologies.
		It ensures all nodes remain "alive" (connected) by validating connectivity.

		Args:
			nodes_per_layer: Dictionary mapping layer index to number of nodes.
							If None, uses self.nodes_per_layer.
			nest_edges_by_layer: Dictionary mapping layer index to number of edges.
								If None, uses self.nest_edges_by_layer.
			seed: Random seed for reproducibility. If None, uses random seed.

		Returns:
			Tuple containing:
				- connectivity_list: List of weight masks for each layer
				- num_attempts: Number of resampling attempts needed to find valid network

		Raises:
			RuntimeError: If valid network cannot be generated after max_attempts.
		"""
		# Use instance variables if not provided
		if nodes_per_layer is None:
			nodes_per_layer = self.nodes_per_layer
		if nest_edges_by_layer is None:
			nest_edges_by_layer = self.nest_edges_by_layer
		
		# Set random seed for reproducibility
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)
		
		connectivity_list: List[torch.Tensor] = []
		num_attempts = 0

		for attempt in range(self.max_attempts):
			if attempt % 1000 == 0 and attempt > 0:
				print(f"  Attempt {attempt}/{self.max_attempts} - still searching for valid network...")

			num_attempts = attempt + 1
			connectivity_list = []
			valid_network = True

			# Generate random weight masks for each layer using uniform random pruning
			for layer_idx in range(len(nodes_per_layer) - 1):
				input_nodes = nodes_per_layer[layer_idx]
				output_nodes = nodes_per_layer[layer_idx + 1]
				total_edges = input_nodes * output_nodes
				desired_edges = nest_edges_by_layer[layer_idx]

				# Create a temporary linear layer and prune it
				temp_linear = torch.nn.Linear(input_nodes, output_nodes)
				# Calculate pruning amount (fraction to prune)
				amount_to_prune = total_edges - desired_edges
				# Prune using random unstructured pruning
				prune.random_unstructured(temp_linear, name="weight", amount=amount_to_prune)

				# Extract the mask from the pruned layer
				# The mask is stored as a buffer named 'weight_mask' after pruning
				mask = getattr(temp_linear, 'weight_mask')  # shape: (out_features, in_features)
				# Transpose to match our expected shape (in_features, out_features)
				weight_mask = mask.T
				connectivity_list.append(weight_mask)

			# Check if all nodes are alive
			if self._check_all_nodes_alive(nodes_per_layer, connectivity_list):
				break
			else:
				valid_network = False

		if not valid_network:
			print(f"❌ Failed to generate valid network after {self.max_attempts} attempts")
			raise RuntimeError(f"Failed to generate valid network after {self.max_attempts} attempts")
		else:
			print(f"✓ Successfully generated valid network after {num_attempts} attempts")

		return connectivity_list, num_attempts

	def build_neural_network(
		self, connectivity_list: List[torch.Tensor], dropout_fraction: float = 0.0
	) -> nn.Sequential:
		"""Build the neural network using connectivity masks.

		Uses torch.prune.custom_from_mask to apply masks, which automatically handles
		mask maintenance during training and works correctly with state_dict loading.
		Masks are applied once at initialization and remain fixed (static sparse network).

		Args:
			connectivity_list: List of weight masks for each layer
			dropout_fraction: Dropout rate for regularization

		Returns:
			Sequential neural network with sparse layers
		"""
		layers: List[nn.Module] = []

		# Add sparse layers using torch.prune to apply masks (static masks)
		for i in range(len(connectivity_list)):
			input_nodes = self.nodes_per_layer[i]
			output_nodes = self.nodes_per_layer[i + 1]
			mask = connectivity_list[i]  # shape: (in, out)
			mask_t = mask.T  # match Linear weight shape (out, in)

			linear = nn.Linear(input_nodes, output_nodes)
			# Use torch.prune.custom_from_mask to apply the weight mask once
			prune.custom_from_mask(linear, name="weight", mask=mask_t)

			layers.append(linear)
			layers.append(self.activation())
			layers.append(nn.BatchNorm1d(output_nodes))
			layers.append(nn.Dropout(p=dropout_fraction))

		# Add final output layer
		final_layer_nodes = self.nodes_per_layer[len(self.nodes_per_layer) - 1]
		layers.append(nn.Linear(final_layer_nodes, 1))
		layers.append(nn.Sigmoid())

		return nn.Sequential(*layers)

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

	def _check_all_nodes_alive(
		self, nodes_per_layer: Dict[int, int], connectivity_list: List[torch.Tensor]
	) -> bool:
		"""Check if every node is alive (has at least one connection).

		A node is considered "alive" if it has at least one incoming connection
		(for output nodes) and at least one outgoing connection (for input nodes).

		Args:
			nodes_per_layer: Dictionary mapping layer index to number of nodes
			connectivity_list: List of weight masks for each layer

		Returns:
			True if all nodes are alive, False otherwise
		"""
		# Check each mask to ensure both dimensions have at least one 1
		for layer_idx, mask in enumerate(connectivity_list):
			# Check dim 0 (rows) - each input node has at least one output connection
			has_output_connections = torch.any(mask, dim=1)
			if not torch.all(has_output_connections):
				return False

			# Check dim 1 (columns) - each output node has at least one input connection
			has_input_connections = torch.any(mask, dim=0)
			if not torch.all(has_input_connections):
				return False

		return True

	def forward(self, X: torch.Tensor, return_auxiliary: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		"""Forward pass through the neural network.

		Similar to eNest implementation: collects activations during forward pass,
		then processes all direct output predictions in parallel at the end.

		Args:
			X: Input tensor of shape (batch_size, input_dim)
			return_auxiliary: If True, also return auxiliary predictions from direct output layers

		Returns:
			Tuple containing:
				- main_output: Main output tensor of shape (batch_size, 1)
				- auxiliary_outputs: List of auxiliary predictions from each intermediary layer
				  Each element is a tensor of shape (batch_size, num_groups) where num_groups
				  is the number of groups in that layer

		Raises:
			RuntimeError: If neural network is not built.
		"""
		if not hasattr(self, "NN") or self.NN is None:
			raise RuntimeError("Neural network not built. This should not happen with automatic building.")

		# Forward pass through main network, collecting activations at each intermediary layer
		intermediary_activations = []  # Collect activations from each intermediary layer
		current_input = X
		
		# We need to manually forward through layers to capture intermediate activations
		# The NN is a Sequential, so we'll iterate through it
		intermediary_layer_idx = 1  # Start from layer 1 (first hidden layer)
		
		# Track which sequential index corresponds to which layer
		sequential_idx = 0
		main_layers = list(self.NN.children())
		
		# Process layers in groups of 4: Linear, Activation, BatchNorm, Dropout
		while sequential_idx < len(main_layers) - 2:  # Exclude final Linear and Sigmoid
			# Linear layer
			linear_layer = main_layers[sequential_idx]
			current_input = linear_layer(current_input)
			sequential_idx += 1
			
			# Activation
			activation_layer = main_layers[sequential_idx]
			current_input = activation_layer(current_input)
			sequential_idx += 1
			
			# BatchNorm
			batchnorm_layer = main_layers[sequential_idx]
			current_input = batchnorm_layer(current_input)
			sequential_idx += 1
			
			# Dropout
			dropout_layer = main_layers[sequential_idx]
			current_input = dropout_layer(current_input)
			sequential_idx += 1
			
			# Collect activation from this intermediary layer (after dropout, before next layer)
			if return_auxiliary and intermediary_layer_idx < len(self.nodes_per_layer) - 1:
				# Store activation for later parallel processing
				intermediary_activations.append(current_input)
			
			intermediary_layer_idx += 1
		
		# Final output layer
		final_linear = main_layers[sequential_idx]
		final_activation = main_layers[sequential_idx + 1]
		main_output = final_activation(final_linear(current_input))
		
		# Process all direct output predictions in parallel (similar to eNest's asm_out approach)
		auxiliary_outputs = []
		if return_auxiliary and self.total_groups > 0 and len(intermediary_activations) > 0:
			# Collect all groups from all layers
			all_groups_flat = []
			group_counts = []  # Track number of groups per layer for splitting output
			
			for layer_idx, layer_activation in enumerate(intermediary_activations):
				layer_mask = self.group_masks[layer_idx]
				if layer_mask is not None:
					batch_size = layer_activation.shape[0]
					num_groups = layer_mask.shape[0]
					
					# Extract groups using mask (similar to eNest's approach)
					# Since groups are consecutive, we can reshape directly
					# layer_activation: (batch_size, num_nodes)
					# Extract first num_groups * genotype_hiddens nodes and reshape
					num_nodes_to_use = num_groups * self.genotype_hiddens
					groups_reshaped = layer_activation[:, :num_nodes_to_use].view(
						batch_size, num_groups, self.genotype_hiddens
					)
					# Flatten: (batch_size, num_groups * genotype_hiddens)
					groups_flat = groups_reshaped.view(batch_size, -1)
					all_groups_flat.append(groups_flat)
					group_counts.append(num_groups)
			
			# Concatenate all groups from all layers: (batch_size, total_groups * genotype_hiddens)
			if all_groups_flat:
				all_groups_concat = torch.cat(all_groups_flat, dim=1)  # (batch_size, total_groups * genotype_hiddens)
				
				# Process all groups in parallel through single linear layer + activation
				all_predictions = self.direct_output_layers(all_groups_concat)  # (batch_size, total_groups)
				
				# Split predictions back by layer
				group_idx = 0
				for num_groups in group_counts:
					# Extract predictions for this layer
					layer_predictions = all_predictions[:, group_idx:group_idx + num_groups]
					auxiliary_outputs.append(layer_predictions)
					group_idx += num_groups
		
		return main_output, auxiliary_outputs

	def finalize_pruning(self) -> None:
		"""Remove pruning re-parametrization and bake masked weights in place.

		Call this before saving the final/eval model to strip mask buffers.
		This permanently removes the pruning masks and keeps only the active weights.
		"""
		for module in self.NN.modules():
			if hasattr(module, "weight_mask"):
				prune.remove(module, "weight")


# ---------------------- Testing ---------------------- #
if __name__ == "__main__":
	print("=== Testing Uniform Random Direct Output Sparse NN Reproducibility ===")
	seed = 42
	genotype_hiddens = 4

	# Test 1: Multiple instances with same seed (auto-build)
	results = []
	for i in range(3):
		sparse_nn = UniformRandomSNN(genotype_hiddens=genotype_hiddens, seed=seed)
		results.append((sparse_nn.num_attempts, sparse_nn.connectivity_list))
		print(f"Instance {i+1}: {sparse_nn.num_attempts} attempts")

	# Check if all instances produced identical results
	attempts = [r[0] for r in results]
	masks = [r[1] for r in results]

	attempts_same = all(a == attempts[0] for a in attempts)
	masks_same = all(
		torch.equal(masks[0][i], masks[j][i]) for j in range(1, len(masks)) for i in range(len(masks[0]))
	)

	print(f"Same attempts: {attempts_same}")
	print(f"Same masks: {masks_same}")
	print(f"✅ Reproducible: {attempts_same and masks_same}")

	# Test 2: Different seeds produce different results
	print("\n=== Testing Different Seeds ===")
	seeds = [42, 123, 456]
	results_diff = []
	for seed in seeds:
		sparse_nn = UniformRandomSNN(genotype_hiddens=genotype_hiddens, seed=seed)
		results_diff.append((sparse_nn.num_attempts, [mask.sum().item() for mask in sparse_nn.connectivity_list]))
		print(f"Seed {seed}: {sparse_nn.num_attempts} attempts, mask sums: {results_diff[-1][1]}")

	# Check if different seeds produced different results
	attempts_diff = len(set(r[0] for r in results_diff)) > 1
	mask_sums_diff = len(set(tuple(r[1]) for r in results_diff)) > 1
	print(f"✅ Different results: {attempts_diff or mask_sums_diff}")

	# Test 3: Test forward pass with auxiliary outputs
	print("\n=== Testing Forward Pass with Direct Output Connections ===")
	sparse_nn_test = UniformRandomSNN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, sparse_nn_test.nodes_per_layer[0])
	try:
		main_output, auxiliary_outputs = sparse_nn_test.forward(input_tensor, return_auxiliary=True)
		print(f"Forward pass successful! Main output shape: {main_output.shape}")
		print(f"Number of intermediary layers with direct output connections: {len(auxiliary_outputs)}")
		for i, aux_out in enumerate(auxiliary_outputs):
			print(f"  Layer {i+1} auxiliary output shape: {aux_out.shape}")
		print(f"✅ Network is ready to use: {sparse_nn_test.NN is not None}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
		import traceback
		traceback.print_exc()

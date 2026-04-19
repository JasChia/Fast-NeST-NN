"""ERK_SNN: Erdos-Renyi Kernel Sparse Neural Network implementation.

This module implements a sparse neural network using ERK (Erdos-Renyi Kernel) sparsity for initialization
of sparse network topology. Previous analyses with Nest VNN utilize 4 nodes per assembly.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
#This code uses Erdos-Renyi Kernel (ERK) sparsity for sparse network topology
class ERK_SNN(nn.Module):
	"""Sparse Neural Network with Erdos-Renyi Kernel (ERK) sparsity.

	ERK_SNN uses ERK sparsity to create sparse network topologies with
	layer-wise edge probabilities based on layer dimensions. The network uses
	static sparse masks applied via PyTorch pruning (no prune/regrow during training).

	Args:
		input_dim: Input dimension (must be 689)
		dropout_fraction: Dropout rate for regularization
		activation: Activation function class (e.g., nn.Tanh, nn.ReLU)
		genotype_hiddens: Number of nodes per assembly (default: 4)
		seed: Random seed for reproducibility
		max_attempts: Maximum attempts to generate valid network topology
		sparsity: Target overall sparsity of the network. If None, uses sparsity
				 computed from nest_edges_by_layer and fc_edges_by_layer.
	"""

	def __init__(
		self,
		input_dim: int = 689,
		dropout_fraction: float = 0.0,
		activation: nn.Module = nn.Tanh,
		genotype_hiddens: int = 4,
		seed: Optional[int] = None,
		max_attempts: int = 10000,
		sparsity: Optional[float] = None,
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

		# Use provided sparsity or fall back to nest network sparsity
		if sparsity is None:
			# Compute nest network sparsity from nest_edges and fc_edges
			total_fc_edges = sum(
				input_nodes * self.nodes_per_layer[layer_idx + 1]
				for layer_idx, input_nodes in enumerate(self.nodes_per_layer.values())
				if layer_idx < len(self.nodes_per_layer) - 1
			)
			total_nest_edges = sum(self.nest_edges_by_layer.values())
			nest_sparsity = 1.0 - (total_nest_edges / total_fc_edges) if total_fc_edges > 0 else 0.0
			sparsity = nest_sparsity

		# Build the sparse network masks using ERK (Erdos-Renyi Kernel) sparsity for initialization
		self.connectivity_list, self.num_attempts = self.build_sparse_network_masks(seed=seed, sparsity=sparsity)
		# Build a static sparse network (no prune/regrow during training)
		# Masks are applied once at init using PyTorch pruning; they stay fixed.
		self.NN = self.build_neural_network(self.connectivity_list, dropout_fraction=dropout_fraction)

	def build_sparse_network_masks(
		self,
		nodes_per_layer: Optional[Dict[int, int]] = None,
		sparsity: float = None,
		seed: Optional[int] = None,
	) -> Tuple[List[torch.Tensor], int]:
		"""Build sparse network masks using ERK (Erdos-Renyi Kernel) sparsity for initialization.

		ERK sparsity assigns layer-wise edge probabilities based on layer dimensions, then
		samples edges according to these probabilities to achieve target edge counts.
		The method ensures all nodes remain "alive" (connected) by validating connectivity.

		Args:
			nodes_per_layer: Dictionary mapping layer index to number of nodes.
							If None, uses self.nodes_per_layer.
			seed: Random seed for reproducibility. If None, uses random seed.
			sparsity: Target overall sparsity of the network. Must be provided.

		Returns:
			Tuple containing:
				- connectivity_list: List of weight masks for each layer
				- num_attempts: Number of resampling attempts needed to find valid network

		Raises:
			RuntimeError: If valid network cannot be generated after max_attempts.
		"""
		if nodes_per_layer is None:
			nodes_per_layer = self.nodes_per_layer

		assert sparsity is not None, "sparsity must be provided"
		assert sparsity >= 0.0 and sparsity <= 1.0, "sparsity must be between 0.0 and 1.0"

		# Set random seed for reproducibility
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

		# Calculate ERK densities for each layer
		# ERK density formula: d_l = (n_{l-1} + n_l) / (n_{l-1} * n_l)
		erk_densities: Dict[int, float] = {}
		total_fc_edges = 0
		for layer_idx in range(len(nodes_per_layer) - 1):
			input_nodes = nodes_per_layer[layer_idx]
			output_nodes = nodes_per_layer[layer_idx + 1]
			total_edges = input_nodes * output_nodes

			assert(total_edges > 0), "Total edges must be greater than 0"
			erk_density = (input_nodes + output_nodes) / total_edges

			erk_densities[layer_idx] = erk_density
			total_fc_edges += total_edges

		# Scale ERK densities to match overall density (1 - sparsity)
		target_density = 1.0 - sparsity
		weighted_density_sum = sum(
			erk_densities[layer_idx] * nodes_per_layer[layer_idx] * nodes_per_layer[layer_idx + 1]
			for layer_idx in erk_densities.keys()
		)
		assert(total_fc_edges > 0), "Total FC edges must be greater than 0"
		avg_erk_density = weighted_density_sum / total_fc_edges

		# Scale ERK densities to match target overall density
		scale_factor = target_density / avg_erk_density
		for layer_idx in erk_densities:
			erk_densities[layer_idx] = min(1.0, erk_densities[layer_idx] * scale_factor)

		connectivity_list: List[torch.Tensor] = []
		num_attempts = 0

		for attempt in range(self.max_attempts):
			if attempt % 1000 == 0 and attempt > 0:
				print(f"  Attempt {attempt}/{self.max_attempts} - still searching for valid network...")

			num_attempts = attempt + 1
			connectivity_list = []
			valid_network = True

			# Generate ERK-based weight masks for each layer
			for layer_idx in range(len(nodes_per_layer) - 1):
				input_nodes = nodes_per_layer[layer_idx]
				output_nodes = nodes_per_layer[layer_idx + 1]
				total_edges = input_nodes * output_nodes
				# Use ERK density to determine number of edges
				erk_density = erk_densities[layer_idx]
				desired_edges = max(1, int(erk_density * total_edges))  # Ensure at least 1 edge
				desired_edges = min(desired_edges, total_edges)  # Ensure not more than total

				# Randomly select exactly desired_edges edges
				all_indices = torch.arange(total_edges)
				selected_indices = all_indices[torch.randperm(total_edges)[:desired_edges]]
				mask_flat = torch.zeros(total_edges)
				mask_flat[selected_indices] = 1.0

				# Reshape to (input_nodes, output_nodes)
				weight_mask = mask_flat.reshape(input_nodes, output_nodes)
				connectivity_list.append(weight_mask)

			# Check if all nodes are alive
			if self._check_all_nodes_alive(nodes_per_layer, connectivity_list):
				break
			else:
				valid_network = False

		if not valid_network:
			raise RuntimeError(f"Failed to generate valid network after {self.max_attempts} attempts")

		print(f"✓ Successfully generated valid ERK network after {num_attempts} attempts")
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

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""Forward pass through the neural network.

		Args:
			X: Input tensor of shape (batch_size, input_dim)

		Returns:
			Output tensor of shape (batch_size, 1)

		Raises:
			RuntimeError: If neural network is not built.
		"""
		if not hasattr(self, "NN") or self.NN is None:
			raise RuntimeError("Neural network not built. This should not happen with automatic building.")

		return self.NN(X)

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
	print("=== Testing ERK_SNN Reproducibility ===")
	seed = 42
	genotype_hiddens = 4

	# Test 1: Multiple instances with same seed (auto-build)
	results = []
	for i in range(3):
		erk_snn = ERK_SNN(genotype_hiddens=genotype_hiddens, seed=seed)
		results.append((erk_snn.num_attempts, erk_snn.connectivity_list))
		print(f"Instance {i+1}: {erk_snn.num_attempts} attempts")

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
		erk_snn = ERK_SNN(genotype_hiddens=genotype_hiddens, seed=seed)
		results_diff.append((erk_snn.num_attempts, [mask.sum().item() for mask in erk_snn.connectivity_list]))
		print(f"Seed {seed}: {erk_snn.num_attempts} attempts, mask sums: {results_diff[-1][1]}")

	# Check if different seeds produced different results
	attempts_diff = len(set(r[0] for r in results_diff)) > 1
	mask_sums_diff = len(set(tuple(r[1]) for r in results_diff)) > 1
	print(f"✅ Different results: {attempts_diff or mask_sums_diff}")

	# Test 3: Test forward pass
	print("\n=== Testing Forward Pass ===")
	erk_snn_test = ERK_SNN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, erk_snn_test.nodes_per_layer[0])
	try:
		output = erk_snn_test.forward(input_tensor)
		print(f"Forward pass successful! Output shape: {output.shape}")
		print(f"✅ Network is ready to use: {erk_snn_test.NN is not None}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")

from torch import nn
#from training_data_wrapper import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
import random

#Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
class Sparse_NN(nn.Module):
	def __init__(self, input_dim=689, dropout_fraction=0.0, activation=nn.Tanh, genotype_hiddens=4, seed=None, max_attempts=10000):
		super().__init__()
		
		assert(input_dim == 689), "input_dim must be 689"
		# Assert genotype_hiddens is an integer
		assert isinstance(genotype_hiddens, int), "genotype_hiddens must be an integer"
		assert genotype_hiddens > 0, "genotype_hiddens must be positive"
		
		# Set activation function
		self.activation = activation
		#self.dropout_fraction = data_wrapper.dropout_fraction
		#self.output_dim = len(data_wrapper.gene_id_mapping)
		#6 nodes per assembly
		self.genotype_hiddens = genotype_hiddens
		self.max_attempts = max_attempts
		self.nest_edges_by_layer = {
			0: 1321 * self.genotype_hiddens, # 7926,5284
			1: 92 * self.genotype_hiddens * self.genotype_hiddens, #3312, 1472
			2: 36 * self.genotype_hiddens * self.genotype_hiddens, #1296, 576
			3: 13 * self.genotype_hiddens * self.genotype_hiddens, # 468, 208
			4: 4 * self.genotype_hiddens * self.genotype_hiddens, # 144, 64
			5: 2 * self.genotype_hiddens * self.genotype_hiddens, # 72, 32
			6: 2 * self.genotype_hiddens * self.genotype_hiddens, #Full, 72, 32
			7: 1 * self.genotype_hiddens * self.genotype_hiddens, #Full, 36, 16
		}
		self.nodes_per_layer = {
			0: input_dim, #Input layer, should be 689!
			1: 76 * self.genotype_hiddens,
			2: 32 * self.genotype_hiddens,
			3: 13 * self.genotype_hiddens,
			4: 4 * self.genotype_hiddens,
			5: 2 * self.genotype_hiddens,
			6: 2 * self.genotype_hiddens,
			7: 1 * self.genotype_hiddens,
			8: 1 * self.genotype_hiddens,
		}
		self.fc_edges_by_layer = {
			0: 689 * 76 * self.genotype_hiddens,
			1: 76 * self.genotype_hiddens * 32 * self.genotype_hiddens,
			2: 32 * self.genotype_hiddens * 13 * self.genotype_hiddens,
			3: 13 * self.genotype_hiddens * 4 * self.genotype_hiddens,
			4: 4 * self.genotype_hiddens * 2 * self.genotype_hiddens,
			5: 2 * self.genotype_hiddens * 2 * self.genotype_hiddens,
			6: 2 * self.genotype_hiddens * 1 * self.genotype_hiddens,
			7: 1 * self.genotype_hiddens * 1 * self.genotype_hiddens
		}
		for k in self.nest_edges_by_layer:
			assert self.nest_edges_by_layer[k] <= self.fc_edges_by_layer[k], f"nest_edges_by_layer[{k}] > fc_edges_by_layer[{k}]"
		
		# Build the sparse neural network
		self.connectivity_list, self.num_attempts = self.build_sparse_network_masks(seed=seed)
		self.NN = self.build_neural_network(
			self.connectivity_list, 
			dropout_fraction=dropout_fraction, 
		)

	def build_sparse_network_masks(self, nodes_per_layer=None, nest_edges_by_layer=None, seed=None):
		"""
		Build a sparse neural network with specified nodes per layer and edge counts using torch.prune.
		
		Args:
			nodes_per_layer (dict, optional): Dictionary mapping layer index to number of nodes.
											  If None, uses self.nodes_per_layer.
			nest_edges_by_layer (dict, optional): Dictionary mapping layer index to number of edges.
												  If None, uses self.nest_edges_by_layer.
			seed (int, optional): Random seed for reproducibility. If None, uses random seed.
			
		Returns:
			tuple: (connectivity_list, num_attempts) where connectivity_list contains
				   the weight masks for each layer and num_attempts is the number of
				   resampling attempts needed to find a valid network
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
		connectivity_list = []
		num_attempts = 0
		
		for attempt in range(self.max_attempts):
			if attempt % 1000 == 0 and attempt > 0:  # Print every 1k attempts
				print(f"  Attempt {attempt}/{self.max_attempts} - still searching for valid network...")
			num_attempts = attempt + 1
			connectivity_list = []
			valid_network = True
			
			# Generate random weight masks for each layer
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
	
	def build_neural_network(self, connectivity_list, dropout_fraction=0.0):
		"""
		Build the actual neural network using the connectivity masks.
		Uses torch.prune.custom_from_mask to apply masks, which automatically handles
		mask maintenance during training and works correctly with state_dict loading.
		
		Args:
			connectivity_list (list): List of weight masks for each layer from build_sparse_network
			dropout_fraction (float): Dropout rate for regularization
			
		Returns:
			torch.nn.Sequential: The built neural network
		"""
		layers = []
		
		# Add sparse layers using torch.prune to apply masks
		for i in range(len(connectivity_list)):
			input_nodes = self.nodes_per_layer[i]
			output_nodes = self.nodes_per_layer[i + 1]
			mask = connectivity_list[i]  # shape: (in, out)
			mask_t = mask.T  # match Linear weight shape (out, in)
			
			linear = torch.nn.Linear(input_nodes, output_nodes)
			# Use torch.prune.custom_from_mask to apply the weight mask
			# This automatically handles mask maintenance during training and state_dict
			prune.custom_from_mask(linear, name="weight", mask=mask_t)
			
			layers.append(linear)
			layers.append(self.activation())
			layers.append(torch.nn.BatchNorm1d(output_nodes))
			layers.append(torch.nn.Dropout(p=dropout_fraction))
		
		# Add final output layer
		layers.append(torch.nn.Linear(self.nodes_per_layer[len(self.nodes_per_layer)-1], 1))
		layers.append(nn.Sigmoid())
		
		return torch.nn.Sequential(*layers)
	
	
	def _check_all_nodes_alive(self, nodes_per_layer, connectivity_list):
		"""
		Check if every node is alive (has at least one connection to deeper and upper layers).
		
		Args:
			nodes_per_layer (dict): Dictionary mapping layer index to number of nodes
			connectivity_list (list): List of weight masks for each layer
			
		Returns:
			bool: True if all nodes are alive, False otherwise
		"""
		num_layers = len(nodes_per_layer)
		
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

	def forward(self, X):
		"""
		Forward pass through the neural network.
		
		Args:
			X (torch.Tensor): Input tensor of shape (batch_size, input_dim)
			
		Returns:
			torch.Tensor: Output tensor of shape (batch_size, 1)
		"""
		if not hasattr(self, 'NN') or self.NN is None:
			raise RuntimeError("Neural network not built. This should not happen with automatic building.")
		
		output = self.NN(X)
		return output

if __name__ == "__main__":
	# Test reproducibility across multiple instances with same seed
	print("=== Testing Reproducibility ===")
	seed = 42
	genotype_hiddens = 4
	
	# Test 1: Multiple instances with same seed (auto-build)
	results = []
	for i in range(3):
		sparse_nn = Sparse_NN(genotype_hiddens=genotype_hiddens, seed=seed)
		results.append((sparse_nn.num_attempts, sparse_nn.connectivity_list))
		print(f"Instance {i+1}: {sparse_nn.num_attempts} attempts")
	
	# Check if all instances produced identical results
	attempts = [r[0] for r in results]
	masks = [r[1] for r in results]
	
	attempts_same = all(a == attempts[0] for a in attempts)
	masks_same = all(torch.equal(masks[0][i], masks[j][i]) for j in range(1, len(masks)) for i in range(len(masks[0])))
	
	print(f"Same attempts: {attempts_same}")
	print(f"Same masks: {masks_same}")
	print(f"✅ Reproducible: {attempts_same and masks_same}")
	
	# Test 2: Different seeds produce different results
	print("\n=== Testing Different Seeds ===")
	seeds = [42, 123, 456]
	results_diff = []
	for seed in seeds:
		sparse_nn = Sparse_NN(genotype_hiddens=genotype_hiddens, seed=seed)
		results_diff.append((sparse_nn.num_attempts, [mask.sum().item() for mask in sparse_nn.connectivity_list]))
		print(f"Seed {seed}: {sparse_nn.num_attempts} attempts, mask sums: {results_diff[-1][1]}")
	
	# Check if different seeds produced different results
	attempts_diff = len(set(r[0] for r in results_diff)) > 1
	mask_sums_diff = len(set(tuple(r[1]) for r in results_diff)) > 1
	print(f"✅ Different results: {attempts_diff or mask_sums_diff}")
	
	# Test 3: Test forward pass
	print("\n=== Testing Forward Pass ===")
	sparse_nn_test = Sparse_NN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, sparse_nn_test.nodes_per_layer[0])
	try:
		output = sparse_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Output shape: {output.shape}")
		print(f"✅ Network is ready to use: {sparse_nn_test.NN is not None}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
	
from torch import nn
#from training_data_wrapper import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
import random

#Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
class GlobalPrunedFC_NN(nn.Module):
	def __init__(self, input_dim=689, dropout_fraction=0.0, activation=nn.Tanh, genotype_hiddens=4, seed=None, use_batchnorm=True):
		super().__init__()
		
		assert(input_dim == 689), "input_dim must be 689"
		# Assert genotype_hiddens is an integer
		assert isinstance(genotype_hiddens, int), "genotype_hiddens must be an integer"
		assert genotype_hiddens > 0, "genotype_hiddens must be positive"
		
		# Set activation function
		self.activation = activation
		self.use_batchnorm = use_batchnorm
		#self.dropout_fraction = data_wrapper.dropout_fraction
		#self.output_dim = len(data_wrapper.gene_id_mapping)
		#6 nodes per assembly
		self.genotype_hiddens = genotype_hiddens
		
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
		self.total_nest_edges = sum(self.nest_edges_by_layer.values())
		# Initialize total FC edges (sum of all fully connected edges)
		# Track current number of nonzero weights (initialized to fully connected)
		self.current_nonzero_weights = sum(self.fc_edges_by_layer.values())
		
		for k in self.nest_edges_by_layer:
			assert self.nest_edges_by_layer[k] <= self.fc_edges_by_layer[k], f"nest_edges_by_layer[{k}] > fc_edges_by_layer[{k}]"
		

		# Build a fully connected neural network using nodes_per_layer
		layers = []
		# Add fully connected layers according to nodes_per_layer
		for i in range(len(self.nodes_per_layer) - 1):
			input_nodes = self.nodes_per_layer[i]
			output_nodes = self.nodes_per_layer[i + 1]
			linear = torch.nn.Linear(input_nodes, output_nodes)
			layers.append(linear)
			layers.append(self.activation())
			if self.use_batchnorm:
				layers.append(torch.nn.BatchNorm1d(output_nodes))
			layers.append(torch.nn.Dropout(p=dropout_fraction))
		
		# Add final output layer
		layers.append(torch.nn.Linear(self.nodes_per_layer[len(self.nodes_per_layer)-1], 1))
		layers.append(nn.Sigmoid())
		
		self.NN = torch.nn.Sequential(*layers)
	
	def prune(self, sparsity_level=0.5):
		"""
		Prune the neural network using global unstructured pruning with L1 norm.
		Applies overall sparsity across all linear layers using prune.global_unstructured.
		
		Args:
			sparsity_level (float): Target sparsity level as percentage (0.0 to 1.0) of current nonzero weights
		"""
		# Collect all linear layers (excluding final output layer)
		parameters_to_prune = []
		layer_idx = 0
		
		for layer in self.NN:
			if isinstance(layer, torch.nn.Linear):
				# Skip the final output layer (it's not in nest_edges_by_layer)
				if layer_idx >= len(self.nest_edges_by_layer):
					layer_idx += 1
					continue
				
				parameters_to_prune.append((layer, 'weight'))
				layer_idx += 1
		
		# Calculate amount to prune as integer based on current nonzero weights
		# sparsity_level is a percentage of current nonzero weights
		# Use sparsity_level as percentage of current nonzero weights
		amount_to_prune = int(self.current_nonzero_weights * sparsity_level)
		# Ensure we don't prune more than needed to reach target
		min_remaining = self.total_nest_edges
		max_to_prune = max(0, self.current_nonzero_weights - min_remaining)
		amount_to_prune = min(amount_to_prune, max_to_prune)
		
		# Only prune if there are weights to prune and layers to prune from
		if amount_to_prune > 0 and len(parameters_to_prune) > 0:
			# Use global unstructured pruning with L1 norm
			prune.global_unstructured(
				parameters_to_prune,
				pruning_method=prune.L1Unstructured,
				amount=amount_to_prune
			)
			
			# Update tracking attribute by counting actual nonzero weights
			self.current_nonzero_weights -= amount_to_prune

	def is_fully_pruned_to_target(self):
		"""
		Check if the total number of nonzero parameters in linear layers equals the total number of NeST parameters.
		
		Returns:
			bool: True if total nonzero weights across all linear layers equals total_nest_edges, False otherwise
		"""
		layer_idx = 0
		total_nonzero_weights = 0
		
		for layer in self.NN:
			if isinstance(layer, torch.nn.Linear):
				# Skip the final output layer (it's not in nest_edges_by_layer)
				if layer_idx >= len(self.nest_edges_by_layer):
					layer_idx += 1
					continue
				
				# Count nonzero edges in this layer
				if hasattr(layer, 'weight_mask'):
					# If pruned, count active (non-zero) weights using the mask
					nonzero_count = int((layer.weight_mask != 0).sum().item())
				else:
					# If not pruned, count nonzero weights directly
					nonzero_count = int((layer.weight != 0).sum().item())
				
				total_nonzero_weights += nonzero_count
				layer_idx += 1
		
		# Check if total matches target
		return total_nonzero_weights == self.total_nest_edges

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
	# Simple FC forward pass test
	print("=== Testing Reproducibility ===")
	seed = 42
	genotype_hiddens = 4
	
	print("\n=== Testing Forward Pass ===")
	fc_nn_test = GlobalPrunedFC_NN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, fc_nn_test.nodes_per_layer[0])
	try:
		output = fc_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Output shape: {output.shape}")
		print(f"✅ Network is ready to use: {fc_nn_test.NN is not None}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
	
from torch import nn
#from training_data_wrapper import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
import random

#Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
class LayerPrunedFC_NN(nn.Module):
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
		Prune the neural network based on L1 norm of weights using PyTorch's prune.l1_unstructured.
		For each layer i, ensures at least self.nest_edges_by_layer[i] edges remain.
		Supports iterative pruning by checking existing masks.
		
		Args:
			sparsity_level (float): Target sparsity level (fraction of weights to prune)
		"""
		layer_idx = 0  # Track which layer index we're on (for nest_edges_by_layer mapping)
		
		for layer in self.NN:
			if isinstance(layer, torch.nn.Linear):
				# Skip the final output layer (it's not in nest_edges_by_layer)
				if layer_idx >= len(self.nest_edges_by_layer):
					continue
				
				# Check if layer has already been pruned
				weight_left_count = self.fc_edges_by_layer[layer_idx]
				if hasattr(layer, 'weight_mask'):
					# Count how many weights are already masked (pruned)
					weight_left_count = int(layer.weight_mask.sum().item())
				
				target_edges = self.nest_edges_by_layer[layer_idx]
				
				edges_to_prune_additional = max(0, min(weight_left_count - target_edges, int(sparsity_level * weight_left_count)))
				# Only prune if we need to remove more edges
				if edges_to_prune_additional > 0:
					# Use PyTorch's l1_unstructured pruning
					# amount can be either float (proportion) or int (absolute number)
					# If already pruned, we need to prune the additional amount
					prune.l1_unstructured(layer, name='weight', amount=edges_to_prune_additional)
				
				layer_idx += 1

	def is_fully_pruned_to_target(self):
		"""
		Check if all layers have exactly the target number of nonzero edges specified in nest_edges_by_layer.
		
		Returns:
			bool: True if all layers have exactly nest_edges_by_layer[i] nonzero edges, False otherwise
		"""
		layer_idx = 0
		
		for layer in self.NN:
			if isinstance(layer, torch.nn.Linear):
				# Skip the final output layer (it's not in nest_edges_by_layer)
				if layer_idx >= len(self.nest_edges_by_layer):
					continue
				
				# Get the target number of edges for this layer
				target_edges = self.nest_edges_by_layer[layer_idx]
				
				# Count nonzero edges in this layer
				if hasattr(layer, 'weight_mask'):
					# If pruned, count active (non-zero) weights using the mask
					nonzero_count = int((layer.weight_mask != 0).sum().item())
				else:
					# If not pruned, count nonzero weights directly
					nonzero_count = int((layer.weight != 0).sum().item())
				
				# Check if this layer matches the target exactly
				if nonzero_count != target_edges:
					return False
				
				layer_idx += 1
		
		# All layers matched their targets
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
	# Simple FC forward pass test
	print("=== Testing Reproducibility ===")
	seed = 42
	genotype_hiddens = 4
	
	print("\n=== Testing Forward Pass ===")
	fc_nn_test = PFC_NN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, fc_nn_test.nodes_per_layer[0])
	try:
		output = fc_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Output shape: {output.shape}")
		print(f"✅ Network is ready to use: {fc_nn_test.NN is not None}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
	
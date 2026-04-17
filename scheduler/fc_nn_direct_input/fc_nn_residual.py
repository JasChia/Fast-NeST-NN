from torch import nn
#from training_data_wrapper import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import random

#Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
class FC_NN_Residual(nn.Module):
	def __init__(self, input_dim=689, dropout_fraction=0.0, activation=nn.Tanh, genotype_hiddens=4, seed=None):
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
		# no sparse attempts needed for fully connected network
		self.input_dim = input_dim
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
		# Build a fully connected neural network using nodes_per_layer
		# Each layer (except input) will receive previous output + original input concatenated
		self.layer_blocks = self.build_neural_network(
			dropout_fraction=dropout_fraction,
		)
		# Build final output layer
		self.output_layer = torch.nn.Linear(self.nodes_per_layer[len(self.nodes_per_layer)-1], 1)
		self.output_activation = nn.Sigmoid()

	def build_neural_network(self, dropout_fraction=0.0):
		"""
		Build a fully connected neural network defined by nodes_per_layer.
		Each layer (except the first) receives previous output + original input concatenated.
		
		Args:
			dropout_fraction (float): Dropout rate for regularization
			
		Returns:
			torch.nn.ModuleList: List of layer blocks
		"""
		layer_blocks = nn.ModuleList()
		
		# First layer: only receives input (no concatenation)
		first_input_nodes = self.nodes_per_layer[0]
		first_output_nodes = self.nodes_per_layer[1]
		first_block = nn.Sequential(
			torch.nn.Linear(first_input_nodes, first_output_nodes),
			self.activation(),
			torch.nn.BatchNorm1d(first_output_nodes),
			torch.nn.Dropout(p=dropout_fraction)
		)
		layer_blocks.append(first_block)
		
		# Remaining layers: receive previous output + original input concatenated
		for i in range(1, len(self.nodes_per_layer) - 1):
			# Input dimension = previous layer output + original input dimension
			input_nodes = self.nodes_per_layer[i] + self.input_dim
			output_nodes = self.nodes_per_layer[i + 1]
			block = nn.Sequential(
				torch.nn.Linear(input_nodes, output_nodes),
				self.activation(),
				torch.nn.BatchNorm1d(output_nodes),
				torch.nn.Dropout(p=dropout_fraction)
			)
			layer_blocks.append(block)
		
		return layer_blocks
	
	# Sparse-specific helpers removed for fully connected model

	def forward(self, X):
		"""
		Forward pass through the neural network.
		Each layer (except the first) receives previous output + original input concatenated.
		
		Args:
			X (torch.Tensor): Input tensor of shape (batch_size, input_dim)
			
		Returns:
			torch.Tensor: Output tensor of shape (batch_size, 1)
		"""
		if not hasattr(self, 'layer_blocks') or self.layer_blocks is None:
			raise RuntimeError("Neural network not built. This should not happen with automatic building.")
		
		# Store original input for concatenation
		original_input = X
		
		# First layer: only receives input (no concatenation)
		x = self.layer_blocks[0](X)
		
		# Remaining layers: concatenate previous output with original input
		for i in range(1, len(self.layer_blocks)):
			# Concatenate previous layer output with original input
			x = torch.cat([x, original_input], dim=1)
			# Pass through layer block
			x = self.layer_blocks[i](x)
		
		# Final output layer
		output = self.output_activation(self.output_layer(x))
		return output

if __name__ == "__main__":
	# Simple FC forward pass test
	print("=== Testing Reproducibility ===")
	seed = 42
	genotype_hiddens = 4
	
	print("\n=== Testing Forward Pass ===")
	fc_nn_test = FC_NN_Residual(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, fc_nn_test.nodes_per_layer[0])
	try:
		output = fc_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Output shape: {output.shape}")
		print(f"✅ Network is ready to use: {fc_nn_test.layer_blocks is not None}")
		print(f"Number of layer blocks: {len(fc_nn_test.layer_blocks)}")
		print(f"Input dimension: {fc_nn_test.input_dim}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
	
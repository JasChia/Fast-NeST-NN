from torch import nn
#from training_data_wrapper import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import random

#Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
class FC_NN(nn.Module):
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
		self.NN = self.build_neural_network(
			dropout_fraction=dropout_fraction,
		)

	def build_neural_network(self, dropout_fraction=0.0):
		"""
		Build a fully connected neural network defined by nodes_per_layer.
		
		Args:
			dropout_fraction (float): Dropout rate for regularization
			
		Returns:
			torch.nn.Sequential: The built neural network
		"""
		layers = []
		# Add fully connected layers according to nodes_per_layer
		for i in range(len(self.nodes_per_layer) - 1):
			input_nodes = self.nodes_per_layer[i]
			output_nodes = self.nodes_per_layer[i + 1]
			linear = torch.nn.Linear(input_nodes, output_nodes)
			layers.append(linear)
			layers.append(self.activation())
			layers.append(torch.nn.BatchNorm1d(output_nodes))
			layers.append(torch.nn.Dropout(p=dropout_fraction))
		
		# Add final output layer
		layers.append(torch.nn.Linear(self.nodes_per_layer[len(self.nodes_per_layer)-1], 1))
		layers.append(nn.Sigmoid())
		
		return torch.nn.Sequential(*layers)
	
	# Sparse-specific helpers removed for fully connected model

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
	fc_nn_test = FC_NN(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, fc_nn_test.nodes_per_layer[0])
	try:
		output = fc_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Output shape: {output.shape}")
		print(f"✅ Network is ready to use: {fc_nn_test.NN is not None}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
	
from torch import nn
#from training_data_wrapper import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import random

#Previous analyses with Nest VNN utilize 4 nodes per assembly (genotype_hiddens = 4)
# Fully Connected Neural Network with Direct Input (concatenated inputs) and Layer Predictions
class FC_NN_DI_Layer_Pred(nn.Module):
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
		
		# Build final aggregation layer that combines all intermediary predictions and final prediction
		num_layers = len(self.nodes_per_layer)  # Number of hidden layers (excluding input and output)
		# Input: num_layers intermediary predictions + 1 final prediction = num_layers + 1
		self.final_aggregation_layer = torch.nn.Linear(num_layers, 1)
		self.final_sigmoid = nn.Sigmoid()
		
		# Build intermediary prediction layers for each hidden layer
		# Each hidden layer will have a linear layer + activation to predict the final output
		# We need one prediction layer for each layer_block
		# layer_blocks[0] outputs nodes_per_layer[1], layer_blocks[1] outputs nodes_per_layer[2], etc.
		self.layer_prediction_layers = nn.ModuleList()
		# Create prediction layers for all hidden layers
		# layer_blocks has len(self.nodes_per_layer) - 1 entries (indices 0 to len-2)
		# Each layer_block[i] outputs nodes_per_layer[i+1] dimensions
		num_hidden_layers = len(self.nodes_per_layer) - 1  # Number of hidden layers (excluding input and output)
		for i in range(num_hidden_layers):
			# Each prediction layer takes the layer output and predicts final output
			# layer_blocks[i] outputs nodes_per_layer[i+1] dimensions
			layer_pred = nn.Sequential(
				torch.nn.Linear(self.nodes_per_layer[i + 1], 1),
				self.activation()
			)
			self.layer_prediction_layers.append(layer_pred)

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
		Each hidden layer also produces an intermediary prediction of the final output.
		All intermediary predictions and the final prediction are concatenated and passed through
		a final aggregation layer with sigmoid to produce a single output.
		
		Args:
			X (torch.Tensor): Input tensor of shape (batch_size, input_dim)
			
		Returns:
			torch.Tensor: Final aggregated output tensor of shape (batch_size, 1)
		"""
		if not hasattr(self, 'layer_blocks') or self.layer_blocks is None:
			raise RuntimeError("Neural network not built. This should not happen with automatic building.")
		
		# Store original input for concatenation
		original_input = X
		
		# List to store intermediary predictions from each hidden layer
		hidden_layer_preds = []
		
		# First layer: only receives input (no concatenation)
		x = self.layer_blocks[0](X)
		
		# Generate intermediary prediction from first layer's output
		# First prediction layer corresponds to first hidden layer (index 0)
		layer_pred = self.layer_prediction_layers[0](x)
		hidden_layer_preds.append(layer_pred)
		
		# Remaining layers: concatenate previous output with original input
		for i in range(1, len(self.layer_blocks)):
			# Concatenate previous layer output with original input
			x = torch.cat([x, original_input], dim=1)
			# Pass through layer block
			x = self.layer_blocks[i](x)
			
			# Generate intermediary prediction from this layer's output
			# Prediction layer index i corresponds to layer block i (since we have prediction for all hidden layers)
			layer_pred = self.layer_prediction_layers[i](x)
			hidden_layer_preds.append(layer_pred)
		
		# Final output layer
		final_output = self.output_activation(self.output_layer(x))
		
		# Concatenate all intermediary predictions: [batch_size, num_hidden_layers]
		hidden_layer_preds_tensor = torch.cat(hidden_layer_preds, dim=1)
		
		# Concatenate all intermediary predictions and final prediction: [batch_size, num_hidden_layers + 1]
		all_predictions = torch.cat([hidden_layer_preds_tensor, final_output], dim=1)
		
		# Pass through final aggregation layer and sigmoid to get single output
		final_aggregated_output = self.final_sigmoid(self.final_aggregation_layer(all_predictions))
		
		return final_aggregated_output

if __name__ == "__main__":
	# Simple FC forward pass test
	print("=== Testing Reproducibility ===")
	seed = 42
	genotype_hiddens = 4
	
	print("\n=== Testing Forward Pass ===")
	fc_nn_test = FC_NN_DI_Layer_Pred(genotype_hiddens=genotype_hiddens, seed=42)
	input_tensor = torch.randn(5, fc_nn_test.nodes_per_layer[0])
	try:
		final_output = fc_nn_test.forward(input_tensor)
		print(f"Forward pass successful! Final output shape: {final_output.shape}")
		print(f"✅ Network is ready to use: {fc_nn_test.layer_blocks is not None}")
		print(f"Number of layer blocks: {len(fc_nn_test.layer_blocks)}")
		print(f"Number of prediction layers: {len(fc_nn_test.layer_prediction_layers)}")
		print(f"Input dimension: {fc_nn_test.input_dim}")
	except Exception as e:
		print(f"❌ Forward pass failed: {e}")
	
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
import csv
import os
import copy

import util
from training_data_wrapper import *
from drugcell_nn import *
from torch.nn import MSELoss
# from ccc_loss import * #Paper utilizes MSELoss


class VNNTrainer():

	def __init__(self, data_wrapper, test_data=None):
		self.data_wrapper = data_wrapper
		self.train_feature = self.data_wrapper.train_feature
		self.train_label = self.data_wrapper.train_label
		self.val_feature = self.data_wrapper.val_feature
		self.val_label = self.data_wrapper.val_label
		self.test_data = test_data  # Add test data for evaluation


	def evaluate_validation_metrics(self, model, val_feature, val_label, cell_features, batch_size=2000):
		"""
		Evaluate the model on validation data and calculate metrics
		"""
		val_label_gpu = val_label.cuda(self.data_wrapper.cuda)
		
		model.eval()
		val_loader = du.DataLoader(du.TensorDataset(val_feature, val_label), batch_size=batch_size, shuffle=False)
		
		val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
		val_loss = 0
		
		with torch.no_grad():
			for i, (inputdata, labels) in enumerate(val_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))
				
				aux_out_map, _ = model(cuda_features)
				
				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
				
				# Calculate validation loss
				for name, output in aux_out_map.items():
					loss = MSELoss()
					if name == 'final':
						val_loss += loss(output, cuda_labels)
		
		# Calculate validation metrics
		val_corr = util.pearson_corr(val_predict, val_label_gpu)
		val_spearman = util.spearman_corr(val_predict, val_label_gpu)
		val_r2 = util.get_r2_score(val_predict, val_label_gpu)
		
		print("=" * 60)
		print("VALIDATION METRICS:")
		print("Validation Pearson Correlation: {:.4f}".format(val_corr))
		print("Validation Spearman Correlation: {:.4f}".format(val_spearman))
		print("Validation R² Score: {:.4f}".format(val_r2))
		print("Validation Loss: {:.4f}".format(val_loss))
		print("=" * 60)
		
		return {
			'pearson_corr': val_corr,
			'spearman_corr': val_spearman,
			'r2_score': val_r2,
			'loss': val_loss,
			'predictions': val_predict.cpu().numpy(),
			'true_labels': val_label_gpu.cpu().numpy()
		}


	def evaluate_test_metrics(self, model, test_data, cell_features, batch_size=2000):
		"""
		Evaluate the model on test data and calculate metrics
		"""
		if test_data is None:
			print("No test data provided for evaluation")
			return None
			
		# Handle the case where test_data is a tuple of (data, mapping)
		if isinstance(test_data, tuple) and len(test_data) == 2 and isinstance(test_data[1], dict):
			test_feature, test_label = test_data[0]
		else:
			test_feature, test_label = test_data
			
		test_label_gpu = test_label.cuda(self.data_wrapper.cuda)
		
		model.eval()
		test_loader = du.DataLoader(du.TensorDataset(test_feature, test_label), batch_size=batch_size, shuffle=False)
		
		test_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
		
		with torch.no_grad():
			for i, (inputdata, labels) in enumerate(test_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				
				# make prediction for test data
				aux_out_map, _ = model(cuda_features)
				
				if test_predict.size()[0] == 0:
					test_predict = aux_out_map['final'].data
				else:
					test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
		
		# Calculate test metrics
		test_corr = util.pearson_corr(test_predict, test_label_gpu)
		test_spearman = util.spearman_corr(test_predict, test_label_gpu)
		test_r2 = util.get_r2_score(test_predict, test_label_gpu)
		
		print("=" * 60)
		print("TEST METRICS:")
		print("Test Pearson Correlation: {:.4f}".format(test_corr))
		print("Test Spearman Correlation: {:.4f}".format(test_spearman))
		print("Test R² Score: {:.4f}".format(test_r2))
		print("=" * 60)
		
		return {
			'pearson_corr': test_corr,
			'spearman_corr': test_spearman,
			'r2_score': test_r2,
			'predictions': test_predict.cpu().numpy(),
			'true_labels': test_label_gpu.cpu().numpy()
		}


	def load_best_model(self, model_dir):
		"""
		Load the best model from the specified directory
		"""
		best_model_path = os.path.join(model_dir, 'model_best.pt')
		final_model_path = os.path.join(model_dir, 'model_final.pt')
		
		if os.path.exists(best_model_path):
			print(f"Loading best model from {best_model_path}")
			self.model = torch.load(best_model_path)
		elif os.path.exists(final_model_path):
			print(f"Loading final model from {final_model_path}")
			self.model = torch.load(final_model_path)
		else:
			raise FileNotFoundError(f"No model found in {model_dir}")
		
		self.model.cuda(self.data_wrapper.cuda)
		return self.model

	def evaluate_saved_model(self, model_dir):
		"""
		Load and evaluate a saved model on validation and test data
		"""
		# Load the best model
		self.load_best_model(model_dir)
		
		# Evaluate on validation data
		print("\nEvaluating saved model on validation data...")
		val_metrics = self.evaluate_validation_metrics(self.model, self.val_feature, self.val_label, self.data_wrapper.cell_features)
		
		# Save validation results
		if val_metrics:
			np.savetxt(os.path.join(model_dir, 'val_predictions.txt'), val_metrics['predictions'], '%.4e')
			np.savetxt(os.path.join(model_dir, 'val_true_labels.txt'), val_metrics['true_labels'], '%.4e')
			
			# Save validation metrics summary
			with open(os.path.join(model_dir, 'val_metrics.txt'), 'w') as f:
				f.write("Validation Metrics Summary\n")
				f.write("=" * 30 + "\n")
				f.write("Pearson Correlation: {:.4f}\n".format(val_metrics['pearson_corr']))
				f.write("Spearman Correlation: {:.4f}\n".format(val_metrics['spearman_corr']))
				f.write("R² Score: {:.4f}\n".format(val_metrics['r2_score']))
				f.write("Loss: {:.4f}\n".format(val_metrics['loss']))

		# Evaluate on test data
		test_metrics = None
		if self.test_data is not None:
			print("\nEvaluating saved model on test data...")
			test_metrics = self.evaluate_test_metrics(self.model, self.test_data, self.data_wrapper.cell_features)
			
			# Save test results
			if test_metrics:
				np.savetxt(os.path.join(model_dir, 'test_predictions.txt'), test_metrics['predictions'], '%.4e')
				np.savetxt(os.path.join(model_dir, 'test_true_labels.txt'), test_metrics['true_labels'], '%.4e')
				
				# Save test metrics summary
				with open(os.path.join(model_dir, 'test_metrics.txt'), 'w') as f:
					f.write("Test Metrics Summary\n")
					f.write("=" * 30 + "\n")
					f.write("Pearson Correlation: {:.4f}\n".format(test_metrics['pearson_corr']))
					f.write("Spearman Correlation: {:.4f}\n".format(test_metrics['spearman_corr']))
					f.write("R² Score: {:.4f}\n".format(test_metrics['r2_score']))

		# Save all metrics to CSV
		self.save_metrics_to_csv(val_metrics, test_metrics, model_dir)
		
		return val_metrics, test_metrics

	def save_metrics_to_csv(self, val_metrics, test_metrics, model_dir):
		"""
		Save validation and test metrics to a CSV file
		"""
		csv_file = os.path.join(model_dir, 'model_metrics.csv')
		
		with open(csv_file, 'w', newline='') as f:
			writer = csv.writer(f)
			
			# Write header
			writer.writerow(['Metric', 'Validation', 'Test'])
			
			# Write metrics
			if val_metrics and test_metrics:
				writer.writerow(['Pearson Correlation', f"{val_metrics['pearson_corr']:.4f}", f"{test_metrics['pearson_corr']:.4f}"])
				writer.writerow(['Spearman Correlation', f"{val_metrics['spearman_corr']:.4f}", f"{test_metrics['spearman_corr']:.4f}"])
				writer.writerow(['R² Score', f"{val_metrics['r2_score']:.4f}", f"{test_metrics['r2_score']:.4f}"])
				writer.writerow(['Loss', f"{val_metrics['loss']:.4f}", "N/A"])
			elif val_metrics:
				writer.writerow(['Pearson Correlation', f"{val_metrics['pearson_corr']:.4f}", "N/A"])
				writer.writerow(['Spearman Correlation', f"{val_metrics['spearman_corr']:.4f}", "N/A"])
				writer.writerow(['R² Score', f"{val_metrics['r2_score']:.4f}", "N/A"])
				writer.writerow(['Loss', f"{val_metrics['loss']:.4f}", "N/A"])
			elif test_metrics:
				writer.writerow(['Pearson Correlation', "N/A", f"{test_metrics['pearson_corr']:.4f}"])
				writer.writerow(['Spearman Correlation', "N/A", f"{test_metrics['spearman_corr']:.4f}"])
				writer.writerow(['R² Score', "N/A", f"{test_metrics['r2_score']:.4f}"])
				writer.writerow(['Loss', "N/A", "N/A"])
		
		print(f"Metrics saved to: {csv_file}")


	def train_model(self):

		# Set seeds for reproducible neural network initialization
		util.set_seeds(self.data_wrapper.seed)
		
		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)

		epoch_start_time = time.time()
		min_loss = None
		best_model_state = None

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		# Set seeds again for reproducible data loading
		util.set_seeds(self.data_wrapper.seed)
		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=True)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=False)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		optimizer.zero_grad()

		print("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\tgrad_norm\telapsed_time")
		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			_gradnorms = torch.empty(len(train_loader)).cuda(self.data_wrapper.cuda) # tensor for accumulating grad norms from each batch in this epoch

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				for name, output in aux_out_map.items():
					loss = MSELoss()
					if name == 'final':
						total_loss += loss(output, cuda_labels)
					else:
						total_loss += self.data_wrapper.alpha * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				_gradnorms[i] = util.get_grad_norm(self.model.parameters(), 2.0).unsqueeze(0) # Save gradnorm for batch
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0] # Save total gradnorm for epoch
			train_corr = util.pearson_corr(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			val_loss = 0
			for i, (inputdata, labels) in enumerate(val_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				aux_out_map, _ = self.model(cuda_features)

				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
					val_label_gpu = cuda_labels
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
					val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

				for name, output in aux_out_map.items():
					loss = MSELoss()
					if name == 'final':
						val_loss += loss(output, cuda_labels)

			val_corr = util.pearson_corr(val_predict, val_label_gpu)

			epoch_end_time = time.time()
			true_auc = torch.mean(train_label_gpu)
			pred_auc = torch.mean(train_predict)
			print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_corr, total_loss, true_auc, pred_auc, val_corr, val_loss, gradnorms, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

			if min_loss == None:
				min_loss = val_loss
				best_model_state = copy.deepcopy(self.model.state_dict())
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))
			elif min_loss - val_loss > self.data_wrapper.delta:
				min_loss = val_loss
				best_model_state = copy.deepcopy(self.model.state_dict())
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))

		# Load the best model for final evaluation
		if best_model_state is not None:
			print("\nLoading best model for final evaluation...")
			self.model.load_state_dict(best_model_state)
			# Save the best model
			torch.save(self.model, self.data_wrapper.modeldir + '/model_best.pt')
			print("Best model saved as model_best.pt")
		else:
			print("\nNo best model found, using final model...")
		
		# Evaluate on validation data after training is complete
		print("\nEvaluating best model on validation data...")
		val_metrics = self.evaluate_validation_metrics(self.model, self.val_feature, self.val_label, self.data_wrapper.cell_features)
		
		# Save validation results
		if val_metrics:
			np.savetxt(self.data_wrapper.modeldir + '/val_predictions.txt', val_metrics['predictions'], '%.4e')
			np.savetxt(self.data_wrapper.modeldir + '/val_true_labels.txt', val_metrics['true_labels'], '%.4e')
			
			# Save validation metrics summary
			with open(self.data_wrapper.modeldir + '/val_metrics.txt', 'w') as f:
				f.write("Validation Metrics Summary\n")
				f.write("=" * 30 + "\n")
				f.write("Pearson Correlation: {:.4f}\n".format(val_metrics['pearson_corr']))
				f.write("Spearman Correlation: {:.4f}\n".format(val_metrics['spearman_corr']))
				f.write("R² Score: {:.4f}\n".format(val_metrics['r2_score']))
				f.write("Loss: {:.4f}\n".format(val_metrics['loss']))

		# Evaluate on test data after training is complete
		test_metrics = None
		if self.test_data is not None:
			print("\nEvaluating best model on test data...")
			test_metrics = self.evaluate_test_metrics(self.model, self.test_data, self.data_wrapper.cell_features)
			
			# Save test results
			if test_metrics:
				np.savetxt(self.data_wrapper.modeldir + '/test_predictions.txt', test_metrics['predictions'], '%.4e')
				np.savetxt(self.data_wrapper.modeldir + '/test_true_labels.txt', test_metrics['true_labels'], '%.4e')
				
				# Save test metrics summary
				with open(self.data_wrapper.modeldir + '/test_metrics.txt', 'w') as f:
					f.write("Test Metrics Summary\n")
					f.write("=" * 30 + "\n")
					f.write("Pearson Correlation: {:.4f}\n".format(test_metrics['pearson_corr']))
					f.write("Spearman Correlation: {:.4f}\n".format(test_metrics['spearman_corr']))
					f.write("R² Score: {:.4f}\n".format(test_metrics['r2_score']))

		# Save all metrics to CSV
		self.save_metrics_to_csv(val_metrics, test_metrics, self.data_wrapper.modeldir)

		return min_loss

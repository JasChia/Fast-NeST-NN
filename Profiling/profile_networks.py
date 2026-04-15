#!/usr/bin/env python3
"""
Comprehensive Network Performance Profiling Script

Compares eNest vs DrugCellNN performance:
- Forward pass timing
- Training iteration timing (forward + loss + backward)
- Multiple batch sizes (powers of 2: 1, 2, 4, ..., 2048)
- Multiple nodes per assembly values (1-10)
- 100 runs per configuration
- Statistical analysis (mean, std)
- Main paper bar figure per NPA (see PAPER_SPEEDUP_BAR_FIGURE)

Note: eNest outputs a single prediction tensor (batch_size, 1) with no intermediary outputs.
      DrugCellNN outputs auxiliary predictions at each pathway level in addition to the final prediction.

Each npa_* subdirectory includes the main speedup bar chart (see PAPER_SPEEDUP_BAR_FIGURE).
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
# from scipy import stats  # Removed - no longer using t-tests
from tqdm import tqdm

# Local vendored models and nest_vnn utilities (see Profiling/helpers/)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from helpers.eNest import eNest
from helpers.drugcell_nn import DrugCellNN
from helpers import util
from helpers.nest_data_paths import ensure_nest_data_files, nest_data_file_paths, resolve_nest_data_dir

# Primary paper figure: paired bars (Model Inference vs Model Training speedup, fNeST-NN vs NeST-VNN).
# Saved under each npa_<n>/ directory by create_relative_speedup_plot.
PAPER_SPEEDUP_BAR_FIGURE = "speedup_fnest_vs_nest_vnn_bar.png"


class DummyDataWrapper:
    """Dummy data wrapper for DrugCellNN initialization using original nest_vnn code"""
    def __init__(self, cuda_device=None, nodes_per_assembly=1):
        datadir = resolve_nest_data_dir()
        ontfile, gene2idfile = nest_data_file_paths(datadir)
        
        # Load gene mapping using original nest_vnn util
        self.gene_id_mapping = util.load_mapping(gene2idfile, 'genes')
        
        # Load ontology using original nest_vnn method (from TrainingDataWrapper.load_ontology)
        self.load_ontology(ontfile)
        
        # Configuration
        self.num_hiddens_genotype = nodes_per_assembly  # Match eNest nodes_per_assembly
        self.min_dropout_layer = 2
        self.dropout_fraction = 0.0  # No dropout for profiling
        self.alpha = 0.5  # Weight for auxiliary losses (matching training default)
        self.cuda = cuda_device
        
        # Create dummy cell features: (num_cells, num_genes, feature_dim)
        # For profiling, we don't need real data, just proper shapes
        self.cell_features = np.random.randn(100, len(self.gene_id_mapping), 1).astype(np.float32)
        
        print(f"Initialized DummyDataWrapper:")
        print(f"  - Number of genes: {len(self.gene_id_mapping)}")
        print(f"  - Number of terms: {len(self.dG.nodes())}")
        print(f"  - Root term: {self.root}")
        print(f"  - Nodes per assembly (num_hiddens_genotype): {nodes_per_assembly}")
    
    def load_ontology(self, file_name):
        """Load ontology using original nest_vnn TrainingDataWrapper.load_ontology method"""
        import networkx as nx
        import networkx.algorithms.components.connected as nxacc
        import networkx.algorithms.dag as nxadag
        
        dG = nx.DiGraph()
        term_direct_gene_map = {}
        term_size_map = {}
        gene_set = set()
        
        file_handle = open(file_name)
        for line in file_handle:
            line = line.rstrip().split()
            if line[2] == 'default':
                dG.add_edge(line[0], line[1])
            else:
                if line[1] not in self.gene_id_mapping:
                    continue
                if line[0] not in term_direct_gene_map:
                    term_direct_gene_map[line[0]] = set()
                term_direct_gene_map[line[0]].add(self.gene_id_mapping[line[1]])
                gene_set.add(line[1])
        file_handle.close()
        
        for term in dG.nodes():
            term_gene_set = set()
            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]
            deslist = nxadag.descendants(dG, term)
            for child in deslist:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]
            # jisoo
            if len(term_gene_set) == 0:
                print('There is empty terms, please delete term:', term)
                sys.exit(1)
            else:
                term_size_map[term] = len(term_gene_set)
        
        roots = [n for n in dG.nodes if dG.in_degree(n) == 0]
        
        uG = dG.to_undirected()
        connected_subG_list = list(nxacc.connected_components(uG))
        
        print('There are', len(roots), 'roots:', roots[0])
        print('There are', len(dG.nodes()), 'terms')
        print('There are', len(connected_subG_list), 'connected componenets')
        
        if len(roots) > 1:
            print('There are more than 1 root of ontology. Please use only one root.')
            sys.exit(1)
        if len(connected_subG_list) > 1:
            print('There are more than connected components. Please connect them.')
            sys.exit(1)
        
        self.dG = dG
        self.root = roots[0]
        self.term_size_map = term_size_map
        self.term_direct_gene_map = term_direct_gene_map


def create_term_mask(term_direct_gene_map, gene_dim, cuda_id):
    """
    Create term masks for gradient masking using original nest_vnn util.
    Wraps util.create_term_mask to handle CPU mode (cuda_id=None).
    """
    if cuda_id is not None:
        # Use original nest_vnn util.create_term_mask for CUDA
        return util.create_term_mask(term_direct_gene_map, gene_dim, cuda_id)
    else:
        # Handle CPU mode (original util.create_term_mask always calls .cuda())
        term_mask_map = {}
        for term, gene_set in term_direct_gene_map.items():
            mask = torch.zeros(len(gene_set), gene_dim)  # Keep on CPU
            for i, gene_id in enumerate(gene_set):
                mask[i, gene_id] = 1
            term_mask_map[term] = mask
        return term_mask_map


def initialize_enest(nodes_per_assembly, cuda_device=None):
    """Initialize eNest with proper setup including gradient hooks"""
    model = eNest(nodes_per_assembly=nodes_per_assembly, dropout=0.0, activation=nn.Tanh, verbosity=-1)
    
    # Move to device
    if cuda_device is not None:
        model = model.cuda(cuda_device)
    
    # Scale parameters by 0.1 (matching training initialization)
    for param in model.parameters():
        param.data = param.data * 0.1
    
    # Register gradient hooks (CRITICAL for proper training behavior)
    model.register_grad_hooks()
    
    return model


def initialize_drugcell(cuda_device=None, nodes_per_assembly=1):
    """Initialize DrugCellNN with proper setup including gradient masking"""
    data_wrapper = DummyDataWrapper(cuda_device=cuda_device, nodes_per_assembly=nodes_per_assembly)
    model = DrugCellNN(data_wrapper)
    
    # Move to device
    if cuda_device is not None:
        model = model.cuda(cuda_device)
    
    # Create term masks for gradient masking (handles both CPU and CUDA)
    term_mask_map = create_term_mask(
        model.term_direct_gene_map,
        model.gene_dim,
        cuda_device
    )
    
    # Scale parameters by 0.1 and apply term-specific masking
    for name, param in model.named_parameters():
        term_name = name.split('_')[0]
        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1
    
    return model, data_wrapper, term_mask_map


def create_dummy_input_enest(batch_size, cuda_device=None):
    """Create dummy input for eNest: (batch_size, 689)"""
    x = torch.randn(batch_size, 689)
    if cuda_device is not None:
        x = x.cuda(cuda_device)
    return x


def create_dummy_input_drugcell(batch_size, data_wrapper, cuda_device=None):
    """Create dummy input for DrugCellNN: (batch_size, num_genes, feature_dim)"""
    num_genes = len(data_wrapper.gene_id_mapping)
    feature_dim = 1
    x = torch.randn(batch_size, num_genes, feature_dim)
    if cuda_device is not None:
        x = x.cuda(cuda_device)
    return x


def create_dummy_target(batch_size, cuda_device=None):
    """Create dummy target for loss computation: (batch_size, 1)"""
    target = torch.randn(batch_size, 1)
    if cuda_device is not None:
        target = target.cuda(cuda_device)
    return target


def time_forward_pass(model, input_data, num_runs=100):
    """Time forward pass over multiple runs"""
    times = []
    model.eval()  # Set to eval mode to disable dropout
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_data)
        
        # Timed runs
        if input_data.is_cuda:
            torch.cuda.synchronize()
        
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_data)
            if input_data.is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return times


def time_training_iteration(model, input_data, target, loss_fn, term_mask_map=None, num_runs=100, alpha=0.5, model_type='eNest'):
    """
    Time a complete training iteration (forward pass + loss computation + backward pass) over multiple runs.
    
    Args:
        model: The model to profile
        input_data: Input tensor
        target: Target tensor
        loss_fn: Loss function (MSELoss)
        term_mask_map: Gradient mask map for DrugCellNN
        num_runs: Number of runs
        alpha: Weight for auxiliary losses (default 0.5, only used for DrugCellNN)
        model_type: 'eNest' or 'DrugCellNN'
        
    Note:
        eNest outputs a single prediction tensor (batch_size, 1) with no intermediary outputs.
        DrugCellNN outputs (aux_out_map, hidden_embeddings_map) with auxiliary predictions at each pathway level.
    """
    times = []
    batch_size = input_data.size(0)
    
    # BatchNorm requires batch_size >= 2 in training mode
    # For batch size 1, we need to use eval mode for BatchNorm layers
    if batch_size == 1:
        # Set model to train mode but set BatchNorm layers to eval mode
        model.train()
        # Set all BatchNorm layers to eval mode to avoid the batch size requirement
        # This includes standard BatchNorm and custom GroupedBatchNorm1d
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
            # Also handle GroupedBatchNorm1d if it exists
            elif hasattr(module, 'bn_layers'):  # GroupedBatchNorm1d has bn_layers attribute
                for bn_layer in module.bn_layers:
                    if isinstance(bn_layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        bn_layer.eval()
    else:
        model.train()  # Set to train mode
    
    # Warmup
    for _ in range(10):
        output = model(input_data)
        
        # Compute loss matching training behavior
        if model_type == 'eNest':
            # eNest: single prediction output (batch_size, 1)
            # Handle both single tensor output and tuple output (for backward compatibility)
            if isinstance(output, tuple):
                final_pred = output[0]  # Take first element if tuple
            else:
                final_pred = output
            total_loss = loss_fn(final_pred, target)
        else:
            # DrugCellNN: (aux_out_map, hidden_embeddings_map)
            # Training: iterate through aux_out_map, apply loss to each
            aux_out_map = output[0]
            total_loss = 0
            for name, output_tensor in aux_out_map.items():
                if name == 'final':
                    total_loss += loss_fn(output_tensor, target)
                else:
                    total_loss += alpha * loss_fn(output_tensor, target)
        
        total_loss.backward()
        model.zero_grad()
    
    # Timed runs
    if input_data.is_cuda:
        torch.cuda.synchronize()
    
    for _ in range(num_runs):
        start = time.perf_counter()
        
        output = model(input_data)
        
        # Compute loss matching training behavior
        if model_type == 'eNest':
            # eNest: single prediction output (batch_size, 1)
            # Handle both single tensor output and tuple output (for backward compatibility)
            if isinstance(output, tuple):
                final_pred = output[0]  # Take first element if tuple
            else:
                final_pred = output
            total_loss = loss_fn(final_pred, target)
        else:
            # DrugCellNN: (aux_out_map, hidden_embeddings_map)
            # Training: iterate through aux_out_map, apply loss to each
            aux_out_map = output[0]
            total_loss = 0
            for name, output_tensor in aux_out_map.items():
                if name == 'final':
                    total_loss += loss_fn(output_tensor, target)
                else:
                    total_loss += alpha * loss_fn(output_tensor, target)
        
        total_loss.backward()
        
        # Apply gradient masking for DrugCellNN (matching training behavior)
        if term_mask_map is not None:
            for name, param in model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                if param.grad is not None:
                    param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
        
        if input_data.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        model.zero_grad()
        times.append(end - start)
    
    return times


def profile_network(model_type, batch_sizes, num_runs=100, cuda_device=None, nodes_per_assembly=1):
    """Profile a network across different batch sizes"""
    print(f"\n{'='*80}")
    print(f"Profiling {model_type}")
    print(f"{'='*80}")
    
    results = {
        'model': model_type,
        'batch_sizes': [],
        'forward_times': [],
        'training_iteration_times': []
    }
    
    loss_fn = nn.MSELoss()
    
    if model_type == 'eNest':
        # Initialize eNest once
        print("Initializing eNest...")
        model = initialize_enest(nodes_per_assembly=nodes_per_assembly, cuda_device=cuda_device)
        data_wrapper = None
        term_mask_map = None
    else:  # DrugCellNN
        # Initialize DrugCellNN once
        print("Initializing DrugCellNN...")
        model, data_wrapper, term_mask_map = initialize_drugcell(cuda_device=cuda_device, nodes_per_assembly=nodes_per_assembly)
    
    print(f"Model initialized. Starting profiling across {len(batch_sizes)} batch sizes...")
    
    for batch_size in tqdm(batch_sizes, desc=f"Batch sizes"):
        print(f"\n  Batch size: {batch_size}")
        
        # Create inputs
        if model_type == 'eNest':
            input_data = create_dummy_input_enest(batch_size, cuda_device)
        else:
            input_data = create_dummy_input_drugcell(batch_size, data_wrapper, cuda_device)
        
        target = create_dummy_target(batch_size, cuda_device)
        
        # Time forward pass
        print(f"    Timing forward pass ({num_runs} runs)...")
        forward_times = time_forward_pass(model, input_data, num_runs)
        
        # Time training iteration (forward + loss + backward)
        print(f"    Timing training iteration ({num_runs} runs)...")
        # Get alpha from data_wrapper if DrugCellNN, default to 0.5
        alpha = data_wrapper.alpha if data_wrapper is not None and hasattr(data_wrapper, 'alpha') else 0.5
        training_times = time_training_iteration(model, input_data, target, loss_fn, term_mask_map, num_runs, alpha, model_type)
        
        results['batch_sizes'].append(batch_size)
        results['forward_times'].append(forward_times)
        results['training_iteration_times'].append(training_times)
        
        print(f"    Forward:  {np.mean(forward_times)*1000:.3f} ± {np.std(forward_times)*1000:.3f} ms")
        print(f"    Training: {np.mean(training_times)*1000:.3f} ± {np.std(training_times)*1000:.3f} ms")
    
    return results


def save_raw_results(results, output_dir):
    """Save raw timing data to JSON file"""
    output_file = os.path.join(output_dir, f"{results['model']}_raw_times.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'model': results['model'],
        'batch_sizes': results['batch_sizes'],
        'forward_times': [list(times) for times in results['forward_times']],
        'backward_times': [list(times) for times in results['training_iteration_times']]  # Keep key name for compatibility
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nRaw results saved to: {output_file}")


def compute_statistics(results_enest, results_drugcell, output_dir):
    """Compute statistics (mean, std, speedup ratios)"""
    print(f"\n{'='*80}")
    print("Computing Statistics")
    print(f"{'='*80}")
    
    batch_sizes = results_enest['batch_sizes']
    
    summary_data = []
    
    for i, batch_size in enumerate(batch_sizes):
        # eNest statistics
        enest_fwd_times = results_enest['forward_times'][i]
        # Handle both key names (training_iteration_times in memory, backward_times in JSON)
        enest_training_times = results_enest.get('training_iteration_times', results_enest.get('backward_times', []))[i]
        
        # DrugCellNN statistics
        dc_fwd_times = results_drugcell['forward_times'][i]
        # Handle both key names (training_iteration_times in memory, backward_times in JSON)
        dc_training_times = results_drugcell.get('training_iteration_times', results_drugcell.get('backward_times', []))[i]
        
        # Compute means and stds (convert to ms)
        enest_fwd_mean = np.mean(enest_fwd_times) * 1000
        enest_fwd_std = np.std(enest_fwd_times) * 1000
        enest_training_mean = np.mean(enest_training_times) * 1000
        enest_training_std = np.std(enest_training_times) * 1000
        
        dc_fwd_mean = np.mean(dc_fwd_times) * 1000
        dc_fwd_std = np.std(dc_fwd_times) * 1000
        dc_training_mean = np.mean(dc_training_times) * 1000
        dc_training_std = np.std(dc_training_times) * 1000
        
        # Speed ratios (DrugCellNN / eNest)
        fwd_ratio = dc_fwd_mean / enest_fwd_mean if enest_fwd_mean > 0 else float('inf')
        training_ratio = dc_training_mean / enest_training_mean if enest_training_mean > 0 else float('inf')
        
        summary_data.append({
            'batch_size': batch_size,
            'enest_forward_mean_ms': enest_fwd_mean,
            'enest_forward_std_ms': enest_fwd_std,
            'drugcell_forward_mean_ms': dc_fwd_mean,
            'drugcell_forward_std_ms': dc_fwd_std,
            'forward_speedup_ratio': fwd_ratio,
            'enest_training_iteration_mean_ms': enest_training_mean,
            'enest_training_iteration_std_ms': enest_training_std,
            'drugcell_training_iteration_mean_ms': dc_training_mean,
            'drugcell_training_iteration_std_ms': dc_training_std,
            'training_iteration_speedup_ratio': training_ratio,
            # Keep backward_* names for backward compatibility
            'enest_backward_mean_ms': enest_training_mean,
            'enest_backward_std_ms': enest_training_std,
            'drugcell_backward_mean_ms': dc_training_mean,
            'drugcell_backward_std_ms': dc_training_std,
            'backward_speedup_ratio': training_ratio
        })
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Forward Pass (inference):")
        print(f"    eNest:      {enest_fwd_mean:.3f} ± {enest_fwd_std:.3f} ms")
        print(f"    DrugCellNN: {dc_fwd_mean:.3f} ± {dc_fwd_std:.3f} ms")
        print(f"    Ratio (DC/eN): {fwd_ratio:.3f}x")
        print(f"  Training Iteration (forward + loss + backward):")
        print(f"    eNest:      {enest_training_mean:.3f} ± {enest_training_std:.3f} ms")
        print(f"    DrugCellNN: {dc_training_mean:.3f} ± {dc_training_std:.3f} ms")
        print(f"    Ratio (DC/eN): {training_ratio:.3f}x")
    
    # Save to CSV
    df = pd.DataFrame(summary_data)
    output_file = os.path.join(output_dir, 'summary_statistics.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSummary statistics saved to: {output_file}")
    
    return df


def create_relative_speedup_plot(summary_df, output_dir):
    """
    Isolated relative speedup bar chart (publication style).

    Speedup is (NeST-VNN / fNeST-NN) = DrugCellNN_time / eNest_time for each batch size.
    Legend: Model Inference (forward), Model Training (training iteration).
    """
    batch_sizes = summary_df['batch_size'].values
    fwd_speedup = summary_df['forward_speedup_ratio'].values
    training_speedup = summary_df['training_iteration_speedup_ratio'].values

    # Okabe–Ito palette; matches hand-tuned publication exports
    color_inference = '#0072B2'
    color_training = '#E69F00'
    dpi_out = 800

    rc = {
        'font.size': 14,
        'font.weight': 'normal',
        'axes.labelsize': 24,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        x_pos = np.arange(len(batch_sizes))
        width = 0.35

        ax.bar(
            x_pos - width / 2,
            fwd_speedup,
            width,
            label='Model Inference',
            color=color_inference,
            alpha=0.95,
            linewidth=0,
        )
        ax.bar(
            x_pos + width / 2,
            training_speedup,
            width,
            label='Model Training',
            color=color_training,
            alpha=0.95,
            linewidth=0,
        )

        ax.set_title('')
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Speedup of fNeST-NN\nover NeST-VNN', fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(batch_sizes, rotation=45, ha='right')

        legend = ax.legend(frameon=False, fontsize=17, prop={'weight': 'bold', 'size': 17})
        for text in legend.get_texts():
            text.set_fontweight('bold')

        ax.grid(True, alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        output_file = os.path.join(output_dir, PAPER_SPEEDUP_BAR_FIGURE)
        plt.savefig(output_file, dpi=dpi_out, bbox_inches='tight', facecolor='white')
        print(f"Paper figure ({PAPER_SPEEDUP_BAR_FIGURE}) saved to: {output_file}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Profile eNest vs DrugCellNN performance')
    parser.add_argument('--cuda', type=int, default=None, help='CUDA device ID (None for CPU)')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of runs per configuration')
    parser.add_argument('--output-dir', type=str, default='profiling_results', 
                       help='Output directory for results')
    parser.add_argument('--min-batch', type=int, default=1, help='Minimum batch size (2^min_batch)')
    parser.add_argument('--max-batch', type=int, default=13, help='Maximum batch size (2^max_batch, default=13 gives up to 8192)')
    parser.add_argument('--min-npa', type=int, default=4, help='Minimum nodes per assembly (default=1)')
    parser.add_argument('--max-npa', type=int, default=4, help='Maximum nodes per assembly (default=10)')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory with red_ontology.txt and red_gene2ind.txt. '
        'Default: <repo>/Data, or set FNEST_ONTOLOGY_DIR / NEST_VNN_DATA_DIR.',
    )

    args = parser.parse_args()

    if args.data_dir:
        os.environ['FNEST_ONTOLOGY_DIR'] = os.path.abspath(os.path.expanduser(args.data_dir))

    try:
        ensure_nest_data_files()
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    # Create main output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate batch sizes (powers of 2)
    batch_sizes = [2**i for i in range(args.min_batch, args.max_batch + 1)]
    
    # Generate nodes_per_assembly values
    npa_values = list(range(args.min_npa, args.max_npa + 1))
    
    print(f"\n{'='*80}")
    print(f"Network Performance Profiling - Multi-NPA Mode")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Device: {'CUDA ' + str(args.cuda) if args.cuda is not None else 'CPU'}")
    print(f"  - Batch sizes: {batch_sizes}")
    print(f"  - Nodes per assembly: {npa_values}")
    print(f"  - Runs per configuration: {args.num_runs}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - NeST data directory: {resolve_nest_data_dir()}")
    
    # Loop through all nodes_per_assembly values
    for npa in npa_values:
        print(f"\n{'#'*80}")
        print(f"# NODES PER ASSEMBLY: {npa}")
        print(f"{'#'*80}")
        
        # Create subdirectory for this npa
        npa_output_dir = os.path.join(output_dir, f'npa_{npa}')
        os.makedirs(npa_output_dir, exist_ok=True)
        
        # Profile eNest
        results_enest = profile_network(
            'eNest', 
            batch_sizes, 
            num_runs=args.num_runs,
            cuda_device=args.cuda,
            nodes_per_assembly=npa
        )
        save_raw_results(results_enest, npa_output_dir)
        
        # Profile DrugCellNN
        results_drugcell = profile_network(
            'DrugCellNN',
            batch_sizes,
            num_runs=args.num_runs,
            cuda_device=args.cuda,
            nodes_per_assembly=npa
        )
        save_raw_results(results_drugcell, npa_output_dir)
        
        # Compute statistics and main paper figure only
        summary_df = compute_statistics(results_enest, results_drugcell, npa_output_dir)
        create_relative_speedup_plot(summary_df, npa_output_dir)
        
        print(f"\nResults for NPA={npa} saved to: {npa_output_dir}/")
        print(f"  (Main paper figure: {os.path.join(npa_output_dir, PAPER_SPEEDUP_BAR_FIGURE)})")
    
    print(f"\n{'='*80}")
    print("Profiling Complete!")
    print(f"{'='*80}")
    print(f"Results saved under: {output_dir}/ (per-NPA subdirectories only)")


if __name__ == '__main__':
    main()

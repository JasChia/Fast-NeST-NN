#!/usr/bin/env python3
"""
Utility script to train a single Uniform Random Direct Output Sparse NN model and emit timing information for
the expensive sections (data loading, sparse mask generation, each epoch, etc.).

Example (from this directory, data under ``<repo>/Data``):
    python profile_single_model.py \
        -cuda 0 \
        -drug 5 \
        -train_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/true_training_data.txt \
        -val_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/validation_data.txt \
        -test_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/test_data.txt \
        -cell2id ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/D5_cell2ind.txt \
        -ge_data ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/D5_GE_Data.txt \
        -seed 100 \
        -output_dir results/D5/D5_profile \
        -epochs 100 \
        -batch_size_power 4 \
        -dropout 0.3 \
        -lr 1e-3
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uniform_random_snn import UniformRandomSNN


# ----------------------------
# Helper utilities
# ----------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_mapping(mapping_file, mapping_type):
    """Load cell line to index mapping"""
    mapping = {}
    with open(mapping_file) as handle:
        for line in handle:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])
    print(f"Total number of {mapping_type} = {len(mapping)}")
    return mapping


def load_train_data(train_file, val_file, cell2id):
    train_df = pd.read_csv(
        train_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"]
    )
    val_df = pd.read_csv(
        val_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"]
    )

    train_features = [[row[0]] for row in train_df.values]
    train_labels = [[float(row[1])] for row in train_df.values]

    val_features = [[row[0]] for row in val_df.values]
    val_labels = [[float(row[1])] for row in val_df.values]

    return train_features, val_features, train_labels, val_labels


def load_pred_data(test_file, cell2id):
    test_df = pd.read_csv(
        test_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"]
    )
    features = [[[cell2id[row[0]]]] for row in test_df.values]
    labels = [[float(row[2])] for row in test_df.values]
    return features, labels


def build_input_vector(input_tensor, cell_features):
    batch_size = input_tensor.size(0)
    num_genes = cell_features.shape[1]
    feature = np.zeros((batch_size, num_genes))
    for i in range(batch_size):
        cell_id = int(input_tensor[i, 0])
        feature[i] = cell_features[cell_id]
    return torch.from_numpy(feature).float()


def describe_timing_section(name, start_ts, durations):
    duration = time.perf_counter() - start_ts
    durations.append({"section": name, "seconds": duration})
    print(f"[TIMING] {name}: {duration:.2f}s")
    return duration


def human_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{sec:.0f}s"


# ----------------------------
# Profiling runner
# ----------------------------


def main():
    parser = argparse.ArgumentParser(description="Profile a single Uniform Random Direct Output Sparse NN training run")
    parser.add_argument("-cuda", type=int, default=0, help="GPU index to use")
    parser.add_argument("-drug", type=int, default=-1, help="Drug ID for logging")
    parser.add_argument("-train_file", required=True)
    parser.add_argument("-val_file", required=True)
    parser.add_argument("-test_file", required=True)
    parser.add_argument("-cell2id", required=True)
    parser.add_argument("-ge_data", required=True)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-output_dir", type=str, default="./profile_results")
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-batch_size_power", type=int, default=4, help="batch size = 2^power")
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-l1", type=float, default=0.0)
    parser.add_argument("-wd", type=float, default=0.0)
    parser.add_argument(
        "-activation", choices=["Tanh", "ReLU"], default="Tanh", help="Activation func"
    )
    parser.add_argument(
        "-profile_json",
        type=str,
        default=None,
        help="Optional path to dump timing JSON",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("===== Profiling Single Uniform Random Direct Output Sparse NN Run =====")
    print(json.dumps(vars(args), indent=2))

    set_seed(args.seed)

    durations = []
    global_start = time.perf_counter()

    # Device
    if torch.cuda.is_available() and args.cuda < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.cuda}")
        torch.cuda.set_device(device)
        print(f"Using device {device}")
    else:
        device = torch.device("cpu")
        print("CUDA unavailable or index out of range. Using CPU.")

    # Data loading timings
    section_start = time.perf_counter()
    cell2id = load_mapping(args.cell2id, "cell lines")
    describe_timing_section("load_cell2id", section_start, durations)

    section_start = time.perf_counter()
    ge_data = pd.read_csv(args.ge_data, sep=",", header=None).values
    describe_timing_section("load_gene_expression", section_start, durations)

    section_start = time.perf_counter()
    train_features, val_features, train_labels, val_labels = load_train_data(
        args.train_file, args.val_file, cell2id
    )
    describe_timing_section("load_train_val", section_start, durations)

    section_start = time.perf_counter()
    test_features, test_labels = load_pred_data(args.test_file, cell2id)
    describe_timing_section("load_test", section_start, durations)

    # Tensor conversions / build input vectors
    section_start = time.perf_counter()
    train_tensor = torch.Tensor(train_features)
    val_tensor = torch.Tensor(val_features)
    test_tensor = torch.Tensor(test_features)

    train_labels = torch.FloatTensor(train_labels).squeeze()
    val_labels = torch.FloatTensor(val_labels).squeeze()
    test_labels = torch.FloatTensor(test_labels).squeeze()

    describe_timing_section("tensor_conversion", section_start, durations)

    section_start = time.perf_counter()
    train_input = build_input_vector(train_tensor, ge_data)
    val_input = build_input_vector(val_tensor, ge_data)
    test_input = build_input_vector(test_tensor, ge_data)
    describe_timing_section("build_input_vectors", section_start, durations)

    # Model construction
    activation_cls = nn.Tanh if args.activation == "Tanh" else nn.ReLU

    section_start = time.perf_counter()
    model = UniformRandomSNN(
        dropout_fraction=args.dropout,
        activation=activation_cls,
        genotype_hiddens=4,
        seed=args.seed,
    ).to(device)
    mask_duration = describe_timing_section("build_sparse_network", section_start, durations)
    print(
        f"Generated sparse network in {human_seconds(mask_duration)} "
        f"(attempts: {getattr(model, 'num_attempts', 'N/A')})"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-05,
        weight_decay=args.wd,
    )
    loss_fn = nn.MSELoss()

    batch_size = int(2 ** args.batch_size_power)
    dataloader = DataLoader(
        TensorDataset(train_input, train_labels.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    epoch_stats = []
    print("\n===== Starting training =====")
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        forward_back_time = 0.0
        batches = 0

        for batch_data, batch_labels in dataloader:
            batches += 1
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            fb_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            preds, _ = model(batch_data, return_auxiliary=False)
            loss = loss_fn(preds, batch_labels)

            if args.l1 > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + args.l1 * l1_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            forward_back_time += time.perf_counter() - fb_start

        epoch_duration = time.perf_counter() - epoch_start

        # Validation timing
        val_start = time.perf_counter()
        with torch.no_grad():
            model.eval()
            val_preds, _ = model(val_input.to(device), return_auxiliary=False)
            val_loss = loss_fn(val_preds, val_labels.unsqueeze(1).to(device)).item()
        val_duration = time.perf_counter() - val_start

        epoch_stats.append(
            {
                "epoch": epoch,
                "batches": batches,
                "epoch_seconds": epoch_duration,
                "forward_backward_seconds": forward_back_time,
                "val_seconds": val_duration,
                "val_loss": val_loss,
            }
        )
        print(
            f"[Epoch {epoch:03d}] "
            f"total={epoch_duration:.2f}s "
            f"fb={forward_back_time:.2f}s "
            f"val={val_duration:.2f}s "
            f"loss={val_loss:.4f}"
        )

    total_duration = describe_timing_section("total_runtime", global_start, durations)
    print(f"\nTraining complete in {human_seconds(total_duration)}")

    summary = {
        "config": vars(args),
        "device": str(device),
        "timings": durations,
        "epoch_stats": epoch_stats,
        "total_seconds": total_duration,
    }

    json_path = args.profile_json or os.path.join(args.output_dir, "profile_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved timing summary to {json_path}")


if __name__ == "__main__":
    main()


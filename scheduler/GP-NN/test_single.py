import torch
import sys
from pathlib import Path

model_path = str(Path(__file__).resolve().parent / "results" / "D5" / "D5_0" / "best_model" / "model_best.pt")

print("Loading model...")
sys.stdout.flush()
sd = torch.load(model_path, map_location='cpu')
print(f"Loaded. Keys: {len(sd)}")

print("\nAll keys and shapes:")
for k, v in sorted(sd.items()):
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
sys.stdout.flush()

# Identify 2D weight tensors (Linear layers)
print("\n2D weight tensors (Linear layers):")
linear_info = []
for k, v in sorted(sd.items()):
    if v.dim() == 2 and ('weight' in k):
        base = k.replace('.weight_orig', '').replace('.weight', '')
        mask_key = base + '.weight_mask'
        if k.endswith('.weight_orig') and mask_key in sd:
            eff = v * sd[mask_key]
            nz = int((eff != 0).sum().item())
            print(f"  {k}: shape={tuple(v.shape)}, nonzero(eff)={nz}/{v.numel()} (has mask)")
        else:
            nz = int((v != 0).sum().item())
            print(f"  {k}: shape={tuple(v.shape)}, nonzero={nz}/{v.numel()}")
        linear_info.append((k, tuple(v.shape), nz, v.numel()))

GENOTYPE_HIDDENS = 4
FC_EDGES = {
    0: 689 * 76 * GENOTYPE_HIDDENS,
    1: 76 * GENOTYPE_HIDDENS * 32 * GENOTYPE_HIDDENS,
    2: 32 * GENOTYPE_HIDDENS * 13 * GENOTYPE_HIDDENS,
    3: 13 * GENOTYPE_HIDDENS * 4 * GENOTYPE_HIDDENS,
    4: 4 * GENOTYPE_HIDDENS * 2 * GENOTYPE_HIDDENS,
    5: 2 * GENOTYPE_HIDDENS * 2 * GENOTYPE_HIDDENS,
    6: 2 * GENOTYPE_HIDDENS * 1 * GENOTYPE_HIDDENS,
    7: 1 * GENOTYPE_HIDDENS * 1 * GENOTYPE_HIDDENS,
}
total_fc = sum(FC_EDGES.values())
print(f"\nTotal FC edges: {total_fc}")

# Exclude the last 2D weight tensor (output layer)
hidden = linear_info[:-1]
total_nz = sum(x[2] for x in hidden)
print(f"Total nonzero in hidden layers: {total_nz}")
print(f"Sparsity (nonzero/fc_edges): {total_nz / total_fc:.6f}")

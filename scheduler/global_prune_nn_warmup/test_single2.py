import torch
import sys
sys.path.insert(0, '.')
from compute_sparsity import compute_sparsity_from_state_dict, TOTAL_FC_EDGES

model_path = "results/D5/D5_0/best_model/model_best.pt"
sd = torch.load(model_path, map_location='cpu')
total_nz, total_fc, density, per_layer = compute_sparsity_from_state_dict(sd)

print("Per-layer breakdown:")
for i in sorted(per_layer.keys()):
    info = per_layer[i]
    print("  Layer {}: NN.{}, shape={}, nonzero={}/{}, density={:.6f}".format(
        i, info['nn_idx'], info['shape'], info['nonzero'], info['fc_edges'], info['layer_density']))

print("\nTotal nonzero: {}".format(total_nz))
print("Total FC edges: {}".format(total_fc))
print("Density (nonzero/fc_edges): {:.6f}".format(density))

# Cross-check: manually sum nonzero from the state dict
manual_nz = 0
for k in sorted(sd.keys()):
    if k.endswith('.weight_orig') and sd[k].dim() == 2:
        import re
        m = re.match(r'NN\.(\d+)\.', k)
        idx = int(m.group(1))
        mask_key = 'NN.{}.weight_mask'.format(idx)
        eff = sd[k] * sd[mask_key]
        nz = int((eff != 0).sum().item())
        shape = tuple(sd[k].shape)
        if shape != (1, 4):  # exclude output layer
            manual_nz += nz
            print("  [verify] NN.{}: shape={}, nonzero={}".format(idx, shape, nz))

print("\nManual total nonzero (hidden only): {}".format(manual_nz))
print("Match: {}".format(manual_nz == total_nz))

"""Quick sample: 3 models per drug to gauge sparsity range."""
import torch
import re
import sys
import os

RESULTS_DIR = "results"
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
TOTAL_FC = sum(FC_EDGES.values())
print("total_fc_edges = {}".format(TOTAL_FC))


def get_nonzero(model_path):
    sd = torch.load(model_path, map_location='cpu')
    layer_weights = {}
    for key in sd.keys():
        m = re.match(r'NN\.(\d+)\.', key)
        if m is None:
            continue
        idx = int(m.group(1))
        if key.endswith('.weight_orig') and sd[key].dim() == 2:
            mask_key = 'NN.{}.weight_mask'.format(idx)
            if mask_key in sd:
                eff = sd[key] * sd[mask_key]
            else:
                eff = sd[key]
            layer_weights[idx] = int((eff != 0).sum().item())
        elif key.endswith('.weight') and 'NN.{}.weight_orig'.format(idx) not in sd and sd[key].dim() == 2:
            layer_weights[idx] = int((sd[key] != 0).sum().item())

    sorted_idx = sorted(layer_weights.keys())
    hidden_nz = sum(layer_weights[i] for i in sorted_idx[:-1])
    return hidden_nz


drugs = sorted(
    [d for d in os.listdir(RESULTS_DIR) if d.startswith('D')],
    key=lambda x: int(x[1:])
)

for drug in drugs:
    drug_dir = os.path.join(RESULTS_DIR, drug)
    exps = sorted(
        [e for e in os.listdir(drug_dir) if e.startswith(drug + '_')],
        key=lambda x: int(x.split('_')[1])
    )
    samples = [exps[0], exps[len(exps)//2], exps[-1]]
    for exp in samples:
        path = os.path.join(drug_dir, exp, "best_model", "model_best.pt")
        if os.path.exists(path):
            nz = get_nonzero(path)
            print("{}/{}: nonzero={}, density={:.6f}".format(drug, exp, nz, nz / TOTAL_FC))
    sys.stdout.flush()

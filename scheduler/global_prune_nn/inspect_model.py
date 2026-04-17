import torch
import sys
sys.path.insert(0, '/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/global_prune_nn')

sample_best = "results/D5/D5_0/best_model/model_best.pt"
sample_final = "results/D5/D5_0/best_model/model_final.pt"

print("=== model_best.pt keys ===")
sd_best = torch.load(sample_best, map_location='cpu')
for k, v in sd_best.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n=== model_final.pt keys ===")
sd_final = torch.load(sample_final, map_location='cpu')
for k, v in sd_final.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

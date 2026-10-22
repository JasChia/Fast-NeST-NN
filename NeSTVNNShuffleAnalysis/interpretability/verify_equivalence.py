"""Assert the serial driver reproduces upstream calc_scores() byte-for-byte."""
import filecmp, shutil, sys, tempfile, time, types
from pathlib import Path
from rlipp_calculator import RLIPPCalculator
from calc_scores_serial import calc_scores_serial

# Resolve inside this clone: <repo>/NeSTVNNShuffleAnalysis/interpretability/
ROOT = str(Path(__file__).resolve().parents[2])
T = sys.argv[1]            # dir holding hidden/ and predict.txt
drug = sys.argv[2]         # e.g. 5
exp = sys.argv[3]          # e.g. 0
CL = f"{ROOT}/Data/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL"
out = Path(tempfile.mkdtemp())

def args(tag, cpus):
    return types.SimpleNamespace(
        ontology=f"{ROOT}/Data/red_ontology.txt",
        test=f"{CL}/train_test_splits/experiment_{exp}/test_data.txt",
        predicted=f"{T}/predict.txt", gene2idfile=f"{ROOT}/Data/red_gene2ind.txt",
        cell2idfile=f"{CL}/D{drug}_cell2ind.txt", sys_output=str(out / f"rlipp_{tag}.out"),
        gene_output=str(out / f"gene_{tag}.out"), cpu_count=cpus, drug_count=0,
        genotype_hiddens=4, hidden=f"{T}/hidden")

t = time.time(); calc_scores_serial(RLIPPCalculator(args("serial", 1)))
print(f"serial driver          : {time.time()-t:6.1f}s")
t = time.time(); RLIPPCalculator(args("upstream", 1)).calc_scores()   # n_jobs=1 -> in-process
print(f"upstream calc_scores   : {time.time()-t:6.1f}s")

ok = True
for a, b in [("rlipp_serial.out", "rlipp_upstream.out"), ("gene_serial.out", "gene_upstream.out")]:
    same = filecmp.cmp(out / a, out / b, shallow=False)
    print(f"{a:22s} vs {b:22s} : {'BYTE-IDENTICAL' if same else 'DIFFERS'}")
    ok &= same
shutil.rmtree(out)
sys.exit(0 if ok else 1)

"""Serial driver for the VERBATIM idekerlab/nest_vnn RLIPPCalculator.

Why this exists: RLIPPCalculator.calc_scores() dispatches its per-term and
per-gene work through joblib Parallel(backend="multiprocessing"), forking after
the class has already used multiprocessing.Pool in load_all_features().  On this
cluster that wrapper hangs (>43 min without emitting a single row) while the
underlying work takes ~40 s.

This file changes ONLY the executor.  Every number is produced by the unmodified
methods of the vendored class -- exec_lm (PCA -> RidgeCV -> spearman),
calc_term_rlipp (P_rho, C_rho, RLIPP = P_rho/C_rho) and calc_gene_rho.  The
iteration order, drug ordering, headers and format strings replicate
calc_scores() exactly, so the output is byte-identical to the upstream code run
single-threaded.  `verify_equivalence.py` asserts that.
"""
import argparse

from rlipp_calculator import RLIPPCalculator


def calc_scores_serial(self):
    drug_pos_map = self.create_drug_pos_map()
    sorted_drugs = list(self.create_drug_corr_map_sorted(drug_pos_map).keys())[0:self.drug_count]
    feature_map, child_feature_map = self.load_all_features()

    with open(self.rlipp_file, "w") as rlipp_file, open(self.gene_rho_file, "w") as gene_rho_file:
        rlipp_file.write('Term\tP_rho\tP_pval\tC_rho\tC_pval\tRLIPP\n')
        gene_rho_file.write('Gene\tRho\tP_val\n')
        for drug in sorted_drugs:
            for term in self.terms:
                rlipp_file.write(self.calc_term_rlipp(
                    feature_map[term], child_feature_map[term], drug_pos_map[drug], term, drug))
            for gene in self.genes:
                gene_rho_file.write(self.calc_gene_rho(
                    feature_map[gene], drug_pos_map[drug], gene, drug))


def main():
    p = argparse.ArgumentParser(description='RLIPP score calculation (serial executor)')
    p.add_argument('-hidden', type=str); p.add_argument('-ontology', type=str)
    p.add_argument('-test', type=str); p.add_argument('-predicted', type=str)
    p.add_argument('-gene2idfile', type=str); p.add_argument('-cell2idfile', type=str)
    p.add_argument('-sys_output', type=str); p.add_argument('-gene_output', type=str)
    p.add_argument('-cpu_count', type=int, default=1)
    p.add_argument('-drug_count', type=int, default=0)
    p.add_argument('-genotype_hiddens', type=int, default=4)
    args = p.parse_args()
    calc_scores_serial(RLIPPCalculator(args))


if __name__ == '__main__':
    main()

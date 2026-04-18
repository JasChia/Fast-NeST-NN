#!/bin/bash
# Run shuffle assignment analysis for all drugs (same drug IDs as the NeST-VNN shuffle experiments).
# Writes JSON and a text log under shuffle_assignment_analysis/outputs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEST_VNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NEST_VNN_ROOT}/.." && pwd)"
cd "${NEST_VNN_ROOT}"

DATA_ROOT="${NEST_VNN_DATA:-${REPO_ROOT}/Data}"
ONTO_FILE="${DATA_ROOT}/red_ontology.txt"
GENE2ID_FILE="${DATA_ROOT}/red_gene2ind.txt"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

OUTPUT_FILE="${OUT_DIR}/shuffle_analysis_all_drugs.json"
LOG_FILE="${OUT_DIR}/shuffle_assignment_correctness.log"

python3 "${SCRIPT_DIR}/analyze_shuffle_assignments.py" \
    -onto "${ONTO_FILE}" \
    -gene2id "${GENE2ID_FILE}" \
    -base_path "${DATA_ROOT}" \
    -all_drugs \
    -max_experiments 50 \
    -output "${OUTPUT_FILE}" \
    -verbose > "${LOG_FILE}" 2>&1

echo ""
echo "Analysis complete!"
echo "  JSON: ${OUTPUT_FILE}"
echo "  Log:  ${LOG_FILE}"

#!/bin/bash
# Sequentially run every training command in jobs.txt (1200 jobs).
# Prerequisite: cd to nest_vnn/ so ../Data and nest_vnn_logs/ resolve correctly.
#
# Usage:
#   cd /path/to/NeSTVNNShuffleAnalysis/nest_vnn
#   ./nest_vnn_jobs/run_all_jobs.sh
#
# For parallel execution on a cluster, split jobs.txt or use your scheduler.

set -euo pipefail

NEST_VNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${NEST_VNN_ROOT}"
JOBS_FILE="${NEST_VNN_ROOT}/nest_vnn_jobs/jobs.txt"

if [[ ! -f "${JOBS_FILE}" ]]; then
  echo "Missing ${JOBS_FILE}" >&2
  exit 1
fi

echo "Working directory: ${NEST_VNN_ROOT}"
echo "Jobs file: ${JOBS_FILE}"
echo "Press Ctrl+C within 5s to cancel..."
sleep 5

n_ok=0
n_fail=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  # Skip comments and blank lines
  [[ "${line}" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// }" ]] && continue
  echo "----------------------------------------------------------------"
  echo "${line}"
  if eval "${line}"; then
    ((n_ok++)) || true
  else
    echo "FAILED (exit $?)" >&2
    ((n_fail++)) || true
  fi
done < "${JOBS_FILE}"

echo "================================================================"
echo "Finished. Success: ${n_ok}  Failed: ${n_fail}"

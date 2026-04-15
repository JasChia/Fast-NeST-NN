#!/bin/bash
# Helper script to run network profiling (expects to be run from this directory)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Default parameters
CUDA_DEVICE=0
NUM_RUNS=100
NODES_PER_ASSEMBLY=1
OUTPUT_DIR="profiling_results"
DATA_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --cpu)
            CUDA_DEVICE="None"
            shift
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --nodes-per-assembly)
            NODES_PER_ASSEMBLY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda DEVICE] [--cpu] [--num-runs N] [--nodes-per-assembly N] [--output-dir DIR] [--data-dir DIR]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Network Performance Profiling"
echo "========================================"
echo "CUDA Device: $CUDA_DEVICE"
echo "Number of runs: $NUM_RUNS"
echo "Nodes per assembly: $NODES_PER_ASSEMBLY"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$DATA_DIR" ]; then
    echo "NeST data directory: $DATA_DIR"
fi
echo ""

# Build command
# Note: profile_networks.py CLI uses --min-npa/--max-npa (not --nodes-per-assembly); pass-through kept for shell compatibility.
DATA_OPT=""
if [ -n "$DATA_DIR" ]; then
    DATA_OPT=" --data-dir $DATA_DIR"
fi

if [ "$CUDA_DEVICE" = "None" ]; then
    CMD="python profile_networks.py --num-runs $NUM_RUNS --min-npa $NODES_PER_ASSEMBLY --max-npa $NODES_PER_ASSEMBLY --output-dir $OUTPUT_DIR${DATA_OPT}"
else
    CMD="python profile_networks.py --cuda $CUDA_DEVICE --num-runs $NUM_RUNS --min-npa $NODES_PER_ASSEMBLY --max-npa $NODES_PER_ASSEMBLY --output-dir $OUTPUT_DIR${DATA_OPT}"
fi

echo "Running command: $CMD"
echo ""

# Run the profiling script
$CMD

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR/"


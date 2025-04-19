#!/bin/bash

# Base directories
BASE_DIR="/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation"
RESULTS_DIR="$BASE_DIR/results/Llama-3.1-8B"

# Input files that are common across all evaluations
PROMPT_FILE="$BASE_DIR/wikitext_data/sampled_wikitext_data.jsonl"
GOLD_REF_FILE="$BASE_DIR/wikitext_data/gold_ref.jsonl"

# Decoding strategies and their parameters
declare -A DECODING_PARAMS
DECODING_PARAMS=(
    ["assisted"]="2 5"
    # Add other decoding strategies and their parameters here
    # Example: ["greedy"]="0"
    # Example: ["beam"]="2 4 6"
)

# GPU device ID - modify as needed
DEVICE_ID=$CUDA_VISIBLE_DEVICES

# Function to run evaluation for a specific configuration
run_evaluation() {
    local decoding_strategy=$1
    local param_value=$2
    local run_number=$3
    local eval_log=$4
    
    local input_dir="$RESULTS_DIR/wikitext/$decoding_strategy/outputs/${decoding_strategy}_${param_value}/run_${run_number}"
    local completion_file="$input_dir/output_${decoding_strategy}_${param_value}_run_${run_number}.jsonl"
    local output_file="$input_dir/combined_${decoding_strategy}_${param_value}_run_${run_number}.jsonl"
    
    if [ -f "$completion_file" ]; then
        # Add a header for this run in the eval log
        echo "==================================================" >> "$eval_log"
        echo "Evaluation for Run $run_number" >> "$eval_log"
        echo "Decoding Strategy: $decoding_strategy" >> "$eval_log"
        echo "Parameter Value: $param_value" >> "$eval_log"
        echo "Timestamp: $(date)" >> "$eval_log"
        echo "==================================================" >> "$eval_log"
        
        # Run evaluation and capture output
        {
            echo "Processing: $decoding_strategy, param=$param_value, run=$run_number"
            python3 "$BASE_DIR/evaluate.py" \
                --prompt_file "$PROMPT_FILE" \
                --completion_file "$completion_file" \
                --gold_ref_file "$GOLD_REF_FILE" \
                --output_file "$output_file" \
                --device_id $DEVICE_ID
            echo "Evaluation completed for $completion_file"
            echo "" # Empty line for readability
        } >> "$eval_log" 2>&1
        
    else
        echo "Warning: Completion file not found: $completion_file" >> "$eval_log"
    fi
}

# Main execution loop
for decoding_strategy in "${!DECODING_PARAMS[@]}"; do
    params=(${DECODING_PARAMS[$decoding_strategy]})
    
    for param_value in "${params[@]}"; do
        # Create evaluation log file for this configuration
        eval_dir="$RESULTS_DIR/wikitext/$decoding_strategy/outputs/${decoding_strategy}_${param_value}"
        eval_log="$eval_dir/evaluation_results.log"
        
        # Create header in the log file
        echo "Evaluation Results for $decoding_strategy (param=$param_value)" > "$eval_log"
        echo "Started at: $(date)" >> "$eval_log"
        echo "=================================================" >> "$eval_log"
        echo "" >> "$eval_log"
        
        for run_number in {1..2}; do
            run_evaluation "$decoding_strategy" "$param_value" "$run_number" "$eval_log"
        done
        
        # Add summary footer
        echo "=================================================" >> "$eval_log"
        echo "Evaluation completed at: $(date)" >> "$eval_log"
    done
done

echo "All evaluations completed!"

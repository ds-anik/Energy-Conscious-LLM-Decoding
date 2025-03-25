#!/bin/bash
#SBATCH --job-name=wikitext
#SBATCH --output=/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/results/Llama-3.1-8B/%x/assisted/output_%j.out
#SBATCH --error=/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/results/Llama-3.1-8B/%x/assisted/error_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1 

# Create directories for logs and outputs 
LOG_DIR="/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/results/Llama-3.1-8B/${SLURM_JOB_NAME}/assisted"
mkdir -p $LOG_DIR/outputs 

module load cuda12.2/toolkit
module load miniconda3/py310/23.1.0-1 
eval "$(conda shell.bash hook)"
conda activate /home/alireza/D1/miniconda3/envs/vLLM_test_env 
 
echo "SLURM JOB ID is $SLURM_JOB_ID"

# Use the GPU IDs from CUDA_VISIBLE_DEVICES
GPUS=$CUDA_VISIBLE_DEVICES

# Array of runs with different parameters
RUNS=(
    "1st Run"
    "2nd Run"
) 

# Array of temperature values
ASSISTED=(2 5 10) 

# Idle time between runs (in seconds)
IDLE_TIME=30

# Longer idle time between temperature tests (in seconds)
TEMP_IDLE_TIME=100 

# Outer loop for temperatures
for TEMP in "${ASSISTED[@]}"; do
    echo "Starting runs with assisted=${TEMP}" 

    # Loop through runs
    for i in "${!RUNS[@]}"; do
        RUN_PARAMS=${RUNS[i]}
        
        # Create a specific output directory for each temperature and run
        RUN_OUTPUT_DIR="$LOG_DIR/outputs/assisted_${TEMP}/run_$((i+1))"
        mkdir -p $RUN_OUTPUT_DIR 

        # Generate timestamp for this run
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        STDOUT_LOG="$RUN_OUTPUT_DIR/output_${TIMESTAMP}.out"
        STDERR_LOG="$RUN_OUTPUT_DIR/error_${TIMESTAMP}.err"

        ( echo "Running run $((i+1)) with ${RUN_PARAMS} and assisted=${TEMP}"
        
        # Start GPU monitoring for this run
        if command -v nvidia-smi &> /dev/null
        then
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,clocks.applications.memory,clocks.applications.graphics --format=csv,nounits -lms 1000 -i $GPUS > "$RUN_OUTPUT_DIR/gpu_monitor_assisted_${TEMP}_run_$((i+1)).csv" & 
            NVIDIA_SMI_PID=$!
            echo "Started nvidia-smi monitoring with PID $NVIDIA_SMI_PID for run $((i+1))"
        else
            echo "nvidia-smi not found. Unable to start GPU monitoring."
        fi
        
        # Run generate.py with output redirection for this specific run
        python3 generate.py \
                --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
                --decoding_method assisted_decoding \
                --world_size 1 \
                --gpus_per_model 1 \
                --batch_size 1 \
                --max_new_tokens 250 \
                --outfile $RUN_OUTPUT_DIR/output_assisted_${TEMP}_run_$((i+1)).jsonl \
                --infile /home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/data_test/wikitext/Llama2_chat_input.jsonl \
                --prompt_lookup_num_tokens ${TEMP}

        # Stop GPU monitoring for this run
        if [ ! -z "$NVIDIA_SMI_PID" ]; then
            kill $NVIDIA_SMI_PID
            echo "Stopped nvidia-smi monitoring process for run $((i+1))" 
        fi ) > "$STDOUT_LOG" 2> "$STDERR_LOG"

        # Idle time between runs (skip after last run)
        if [ $i -lt $((${#RUNS[@]} - 1)) ]; then
            echo "Waiting for $IDLE_TIME seconds between runs..."
            sleep $IDLE_TIME
        fi 
    done

    # Idle time between temperature tests (skip after last temperature)
    if [ "$TEMP" != "${ASSISTED[-1]}" ]; then
        echo "Waiting for $TEMP_IDLE_TIME seconds between temperature tests..."
        sleep $TEMP_IDLE_TIME
    fi
done 
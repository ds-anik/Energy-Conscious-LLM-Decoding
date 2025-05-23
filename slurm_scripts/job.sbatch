#!/bin/bash
#SBATCH --job-name=translation-en-de
#SBATCH --output=/home/alireza/D1/benchmarking_results/%x/assisted/output_%j.out
#SBATCH --error=/home/alireza/D1/benchmarking_results/%x/assisted/error_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1 

# Create directories for logs and outputs 
LOG_DIR="/home/alireza/D1/benchmarking_results/${SLURM_JOB_NAME}/assisted"
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
    "3rd Run"
    "4th Run"
    "5th Run"
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
        
        echo "Running run $((i+1)) with ${RUN_PARAMS} and assisted=${TEMP}"
        
        # Start GPU monitoring for this run
        if command -v nvidia-smi &> /dev/null
        then
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw --format=csv,nounits -lms 1000 -i $GPUS > "$LOG_DIR/gpu_monitor_assisted_${TEMP}_run_$((i+1)).csv" & 
            NVIDIA_SMI_PID=$!
            echo "Started nvidia-smi monitoring with PID $NVIDIA_SMI_PID for run $((i+1))"
        else
            echo "nvidia-smi not found. Unable to start GPU monitoring."
        fi
        
        # Run lm_eval with specific parameters for each iteration
        lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype="bfloat16",device="cuda",parallelize=False \
                --tasks wmt16-en-de \
                --batch_size 1 \
                --limit 500 \
                --output_path $RUN_OUTPUT_DIR \
                --log_samples \
                --gen_kwargs do_sample=False,temperature=1.0,top_p=1.0,top_k=0,prompt_lookup_num_tokens=${TEMP} 


        # Stop GPU monitoring for this run
        if [ ! -z "$NVIDIA_SMI_PID" ]; then
            kill $NVIDIA_SMI_PID
            echo "Stopped nvidia-smi monitoring process for run $((i+1))" 
        fi 

        # Idle time between runs (skip after last run)
        if [ $i -lt $((${#RUNS[@]} - 1)) ]; then
            echo "Waiting for $IDLE_TIME seconds between runs..."
            sleep $IDLE_TIME
        fi 

    done

    # Idle time between temperature tests (skip after last temperature)
    if [ $TEMP != "${ASSISTED[-1]}" ]; then
        echo "Waiting for $TEMP_IDLE_TIME seconds before starting next assisted decoding runs..."
        sleep $TEMP_IDLE_TIME
    fi

done

echo "Script execution completed."
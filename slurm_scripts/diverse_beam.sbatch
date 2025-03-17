#!/bin/bash
#SBATCH --job-name=code2text-java
#SBATCH --output=/home/alireza/D1/benchmarking_results/%x/diverseBeam/output_%j.out
#SBATCH --error=/home/alireza/D1/benchmarking_results/%x/diverseBeam/error_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1 

# Create directories for logs and outputs 
LOG_DIR="/home/alireza/D1/benchmarking_results/${SLURM_JOB_NAME}/diverseBeam"
mkdir -p $LOG_DIR/outputs 

module load cuda12.2/toolkit
module load miniconda3/py310/23.1.0-1 
eval "$(conda shell.bash hook)"
# conda activate /home/alireza/D1/miniconda3/envs/vLLM_test_env 

source /home/alireza/D1/virtual_envs/code_bench/bin/activate 

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

# Array of beam search parameters (beam size, diversity penalty)
BEAM_PARAMS=(
    "4,2"
    "5,5"
    "10,2"
    "10,5"
)

# Idle time between runs (in seconds)
IDLE_TIME=30

# Longer idle time between temperature tests (in seconds)
TEMP_IDLE_TIME=100 

# Outer loop for temperatures
for PARAM_PAIR in "${BEAM_PARAMS[@]}"; do
    IFS=',' read -r BEAM_SIZE BEAM_GROUP <<< "$PARAM_PAIR"
    echo "Starting runs with beam_size=${BEAM_SIZE} and beam_group_size=${BEAM_GROUP}"

    # Loop through runs
    for i in "${!RUNS[@]}"; do
        RUN_PARAMS=${RUNS[i]}
        
        # Create a specific output directory for each temperature and run
        RUN_OUTPUT_DIR="$LOG_DIR/outputs/diverse_beam_${BEAM_SIZE}_group_${BEAM_GROUP}/run_$((i+1))"
        mkdir -p $RUN_OUTPUT_DIR
        
        echo "Running run $((i+1)) with ${RUN_PARAMS} , beam_size=${BEAM_SIZE}, and beam_group_size=${BEAM_GROUP}"
        
        # Start GPU monitoring for this run
        if command -v nvidia-smi &> /dev/null
        then
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw --format=csv,nounits -lms 1000 -i $GPUS > "$LOG_DIR/gpu_monitor_beam_${BEAM_SIZE}_group_${BEAM_GROUP}_run_$((i+1)).csv" & 
            NVIDIA_SMI_PID=$!
            echo "Started nvidia-smi monitoring with PID $NVIDIA_SMI_PID for run $((i+1))"
        else
            echo "nvidia-smi not found. Unable to start GPU monitoring."
        fi
        
        # Run lm_eval with specific parameters for each iteration
        lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype="bfloat16",device="cuda",parallelize=False \
                --tasks code2text_javascript \
                --batch_size 1 \
                --limit 100 \
                --output_path $RUN_OUTPUT_DIR \
                --log_samples \
                --gen_kwargs do_sample=False,num_beams=${BEAM_SIZE},num_beam_groups=${BEAM_GROUP},diversity_penalty=1.0,temperature=1.0,top_k=0,top_p=1.0,max_new_tokens=250


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
    if [ $PARAM_PAIR != "${BEAM_PARAMS[-1]}" ]; then
        echo "Waiting for $TEMP_IDLE_TIME seconds before starting next assisted decoding runs..."
        sleep $TEMP_IDLE_TIME
    fi

done

echo "Script execution completed."
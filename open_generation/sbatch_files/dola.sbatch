#!/bin/bash
#SBATCH --job-name=wikitext
#SBATCH --output=/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/results/Llama-3.1-8B/%x/dola/output_%j.out
#SBATCH --error=/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/results/Llama-3.1-8B/%x/dola/error_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH -w n013 

# Create directories for logs and outputs 
LOG_DIR="/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/results/Llama-3.1-8B/${SLURM_JOB_NAME}/dola"
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


# Array of DoLA decoding layers
DOLA_LAYERS=(
    "high"
    "low"
)

# Idle time between runs (in seconds)
IDLE_TIME=30

# Longer idle time between temperature tests (in seconds)
TEMP_IDLE_TIME=100 

# Loop through DoLA decoding layers
for DOLA_LAYER in "${DOLA_LAYERS[@]}"; do
    echo "Starting runs with DoLA layer=${DOLA_LAYER}"

    # Loop through runs
    for i in "${!RUNS[@]}"; do
        RUN_PARAMS=${RUNS[i]}
        
        # Create a specific output directory for each temperature and run
        RUN_OUTPUT_DIR="$LOG_DIR/outputs/dola_${DOLA_LAYER}/run_$((i+1))"
        mkdir -p $RUN_OUTPUT_DIR 

        # Generate timestamp for this run
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        STDOUT_LOG="$RUN_OUTPUT_DIR/output_${TIMESTAMP}.out"
        STDERR_LOG="$RUN_OUTPUT_DIR/error_${TIMESTAMP}.err"

        ( echo "Running run $((i+1)) with ${RUN_PARAMS} and dola=${DOLA_LAYER}"
        
        # Start GPU monitoring for this run
        if command -v nvidia-smi &> /dev/null
        then
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,clocks.applications.memory,clocks.applications.graphics --format=csv,nounits -lms 1000 -i $GPUS > "$LOG_DIR/gpu_monitor_dola_${DOLA_LAYER}_run_$((i+1)).csv" & 
            NVIDIA_SMI_PID=$!
            echo "Started nvidia-smi monitoring with PID $NVIDIA_SMI_PID for run $((i+1))"
        else
            echo "nvidia-smi not found. Unable to start GPU monitoring."
        fi
        
        # Run generate.py with output redirection for this specific run
        python3 /home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/generate.py \
                --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
                --decoding_method dola \
                --world_size 1 \
                --gpus_per_model 1 \
                --batch_size 1 \
                --outfile $RUN_OUTPUT_DIR/output_dola_${DOLA_LAYER}_run_$((i+1)).jsonl \
                --infile /home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/wikitext_data/sampled_wikitext_data.jsonl \
                --dola_layers ${DOLA_LAYER}

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
    if [ $DOLA_LAYER != "${DOLA_LAYERS[-1]}" ]; then
        echo "Waiting for $TEMP_IDLE_TIME seconds before starting next dola layer runs..."
        sleep $TEMP_IDLE_TIME
    fi
    
done 
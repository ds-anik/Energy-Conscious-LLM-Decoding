import json
import mauve 
import argparse
from transformers import AutoTokenizer

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def calculate_mauve(full_texts, gold_refs, device_id=0):
    # Convert inputs to lists if they aren't already
    if not isinstance(full_texts, list):
        full_texts = [full_texts]
    if not isinstance(gold_refs, list):
        gold_refs = [gold_refs]
    
    out = mauve.compute_mauve(p_text=full_texts, q_text=gold_refs, device_id=device_id, max_text_length=256, mauve_scaling_factor=1, featurize_model_name='gpt2-xl', verbose=False) 
    return out

def combine_prompt_completion(prompt_file, completion_file, gold_ref_file, output_file, device_id=0):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # Read prompts
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # Extract the actual prompt from the instructions
            # Format: "[INST] Please help me complete the text continuation based on the following content.\n\n{prompt}[/INST]"
            full_instruction = data['instructions']
            prompt = full_instruction.split('\n\n')[1].split('[/INST]')[0].strip()
            prompts.append(prompt)
    
    # Read completions
    completions = []
    with open(completion_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            completions.append(data['completion'])
            
    # Read gold references
    gold_refs = []
    with open(gold_ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gold_refs.append(data['gold_ref'])
    
    # Combine prompts and completions
    combined_outputs = []
    total_tokens = 0
    full_texts = []  # Store full_texts for MAUVE calculation
    
    for idx, (prompt, completion, gold_ref) in enumerate(zip(prompts, completions, gold_refs)):
        full_text = f"{prompt} {completion}"
        num_tokens = count_tokens(full_text, tokenizer)
        total_tokens += num_tokens
        
        combined = {
            'prompt': prompt,
            'completion': completion,
            'full_text': full_text,
            'gold_ref': gold_ref,
            'num_tokens': num_tokens
        }
        combined_outputs.append(combined)
        full_texts.append(full_text)

    
    # Calculate MAUVE score
    print("\nCalculating MAUVE score...")
    mauve_score = calculate_mauve(full_texts, gold_refs, device_id=device_id)
    print(f"MAUVE score: {mauve_score.mauve}")
    
    # Write combined outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in combined_outputs:
            f.write(json.dumps(output) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Combine prompts with their completions and calculate MAUVE score')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt file')
    parser.add_argument('--completion_file', type=str, required=True, help='Path to the completion file')
    parser.add_argument('--gold_ref_file', type=str, required=True, help='Path to the gold reference file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the combined outputs')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID for MAUVE calculation')
    
    args = parser.parse_args()
    
    combine_prompt_completion(args.prompt_file, args.completion_file, args.gold_ref_file, args.output_file, args.device_id)
    print(f"\nSuccessfully combined prompts and completions. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()


    # echo "Generation completed. Running evaluation..."
    
    # # Run evaluate.py to combine outputs and calculate MAUVE score
    # python3 /home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/evaluate.py \
    #         --prompt_file /home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/wikitext_data/sampled_wikitext_data.jsonl \
    #         --completion_file $RUN_OUTPUT_DIR/output_greedy_${TEMP}_run_$((i+1)).jsonl \
    #         --gold_ref_file /home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/wikitext_data/gold_ref.jsonl \
    #         --output_file $RUN_OUTPUT_DIR/combined_greedy_${TEMP}_run_$((i+1)).jsonl \
    #         --device_id $GPUS

    # echo "Evaluation completed."

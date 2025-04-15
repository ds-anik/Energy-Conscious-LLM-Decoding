import json
import random
import os
import re
from transformers import AutoTokenizer

def count_instruction_wrapper_tokens():
    # Initialize the Llama tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # The instruction wrapper parts
    instruction_prefix = "[INST] Please help me complete the text continuation based on the following content.\n\n"
    instruction_suffix = "[/INST]"
    full_wrapper = instruction_prefix + instruction_suffix
    
    # Count tokens
    tokens = tokenizer.encode(full_wrapper)
    print(f"\nInstruction wrapper token analysis:")
    print(f"Total tokens in wrapper: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens]}\n")
    
    return len(tokens)

def is_clean_text(text):
    # Check if the text contains any non-ASCII characters
    return all(ord(char) < 128 for char in text)

def sample_and_format_wikitext(input_file, output_file, gold_ref_file, num_samples=100, random_seed=42):
    # First count the instruction wrapper tokens
    wrapper_tokens = count_instruction_wrapper_tokens()
    print(f"Each prompt will have {wrapper_tokens} additional tokens from the instruction wrapper\n")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Filter and collect clean samples
    clean_samples = []
    for line in lines:
        data = json.loads(line.strip())
        prompt = data.get('instruction', '').strip()  # Get the prompt text
        gold_ref = data.get('gold_ref', '').strip()  # Get the gold reference
        
        # Only keep samples where both prompt and gold_ref are clean
        if is_clean_text(prompt) and is_clean_text(gold_ref):
            clean_samples.append(line)
    
    print(f"Found {len(clean_samples)} clean samples out of {len(lines)} total samples")
    
    if len(clean_samples) < num_samples:
        print(f"Warning: Only {len(clean_samples)} clean samples available, which is less than requested {num_samples}")
        sampled_lines = clean_samples
    else:
        # Sample random lines from clean samples
        sampled_lines = random.sample(clean_samples, num_samples)
    
    # Format each sample with the instruction template
    formatted_samples = []
    gold_references = []
    instruction_prefix = "[INST] Please help me complete the text continuation based on the following content.\n\n"
    instruction_suffix = "[/INST]"
    
    for line in sampled_lines:
        # Parse the original JSON line
        data = json.loads(line.strip())
        prompt = data.get('instruction', '').strip()  # Get the prompt text
        gold_ref = data.get('gold_ref', '').strip()  # Get the gold reference
        
        # Create the formatted sample
        formatted_sample = {
            "instructions": f"{instruction_prefix}{prompt} {instruction_suffix}"
        }
        formatted_samples.append(formatted_sample)
        
        # Store the gold reference
        gold_ref_sample = {
            "gold_ref": gold_ref
        }
        gold_references.append(gold_ref_sample)
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(gold_ref_file), exist_ok=True)
    
    # Write the formatted samples to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in formatted_samples:
            f.write(json.dumps(sample) + '\n')
            
    # Write the gold references to a separate file
    with open(gold_ref_file, 'w', encoding='utf-8') as f:
        for ref in gold_references:
            f.write(json.dumps(ref) + '\n')

if __name__ == "__main__":
    input_file = "/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/wikitext_data/wikitext_data.jsonl"
    output_file = "/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/wikitext_data/sampled_wikitext_data.jsonl"
    gold_ref_file = "/home/alireza/D1/Energy-Conscious-LLM-Decoding/open_generation/wikitext_data/gold_ref.jsonl"
    
    # Use a fixed random seed for reproducibility
    random_seed = 0
    sample_and_format_wikitext(input_file, output_file, gold_ref_file, random_seed=random_seed)
    print(f"Successfully sampled and formatted examples to {output_file}")
    print(f"Gold references saved to {gold_ref_file}")

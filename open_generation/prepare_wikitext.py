import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def prepare_wikitext_dataset(tokenizer_name="meta-llama/Llama-3.1-8B-Instruct", prompt_length=40, min_sequence_length=160, max_sequence_length=250):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load wikitext datasets
    datasets_val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    datasets_test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    datasets = concatenate_datasets([datasets_val, datasets_test])
    
    def tokenize_function(examples):
        # Clean and prepare text
        texts = [x.replace(' <newline>', '\n') for x in examples['text']]
        texts = [tokenizer.bos_token + x for x in texts if len(x) > 0]
        
        # Tokenize
        result_dict = tokenizer(texts, add_special_tokens=False)
        
        # Create prompts and gold sequences
        input_texts = []
        gold_texts = []
        
        for idx, input_ids in enumerate(result_dict['input_ids']):
            if min_sequence_length <= len(input_ids) <= max_sequence_length:
                # Decode the prompt (first `prompt_length` tokens)
                prompt_text = tokenizer.decode(input_ids[:prompt_length], skip_special_tokens=True)
                # Decode the full text as gold reference
                gold_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                
                input_texts.append(prompt_text)
                gold_texts.append(gold_text)
        
        return {'input_text': input_texts, 'gold_text': gold_texts}
    
    # Process the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=datasets.column_names,
        load_from_cache_file=True,
    )
    
    return tokenized_datasets

def save_to_jsonl(tokenized_datasets, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in tokenized_datasets:
            # Create the JSON object
            json_object = {
                "instruction": item['input_text'],  # Prompt (first text)
                "gold_ref": item['gold_text']       # Complete reference (first text)
            }
            # Write the JSON object as a line in the file
            f.write(json.dumps(json_object) + '\n')

# Usage example:
if __name__ == "__main__":
    # Get the processed dataset
    processed_dataset = prepare_wikitext_dataset()
    
    # Save the results to a .jsonl file
    save_to_jsonl(processed_dataset, 'wikitext_data.jsonl')
    
    # Print some information about the dataset
    print(f"Dataset size: {len(processed_dataset)}")
    
    # Example of accessing the first item
    first_item = processed_dataset[0]
    print(f"First prompt: {first_item['input_text']}")
    print(f"First gold sequence: {first_item['gold_text']}")
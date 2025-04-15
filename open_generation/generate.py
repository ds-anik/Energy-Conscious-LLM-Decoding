# Modified version of Code For A Thorough Examination of Decoding Methods in the Era of LLMs
# Original Code: (https://github.com/DavidFanzz/llm_decoding?tab=readme-ov-file#code-for-a-thorough-examination-of-decoding-methods-in-the-era-of-llms) 

import json
import sys
import torch
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import os
import math
import time
import json
from tqdm import tqdm
from multiprocessing import Process
import pandas as pd
import numpy as np
import random



def args_parse():
    parser = argparse.ArgumentParser(description="Text Generation for Different Decoding Methods")
    parser.add_argument("--infile", type=str, help="The input dataset in .json format")
    parser.add_argument("--outfile", type=str, help="The output file where the generated text will be saved")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=270)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--decoding_method", type=str, default="greedy") 
    parser.add_argument("--gpus_per_model", type=int, default=1)
    parser.add_argument("--begin_gpu", default=0, type=int) 
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct", type=str) 

    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float) 
    parser.add_argument("--typical_p", default=0.95, type=float) 
    parser.add_argument("--min_p", default=0.3, type=float)      
    parser.add_argument("--cs_alpha", default=0.6, type=float) 
    parser.add_argument("--cs_k", default=5, type=int)
    parser.add_argument("--epsilon_cutoff", default=3e-4, type=float)
    parser.add_argument("--dola_layers", default="low", help="Layers to use for DoLA decoding. Can be 'low', 'high' or a list ofintegers containing the layer indices") 
    parser.add_argument("--prompt_lookup_num_tokens", default=2, type=int)
    parser.add_argument("--num_beams", default=2, type=int)
    parser.add_argument("--num_beam_groups", default=2, type=int) 

    
    args = parser.parse_args()
    return args


def out_file(outfile_path, generation_lst):
    with open(outfile_path, "w", encoding="utf-8") as f:
        json.dump(generation_lst, f, indent=4)

    print(f"written to {outfile_path}")


def verify_generation_config(model, generation_config, decoding_method, rank=0):
    """Verify that generation config matches our intended parameters."""
    if rank == 0:  # Only print from main process
        print(f"Current generation config:")
        print(f"  Method: {decoding_method}")
        
        # Create a GenerationConfig object from our kwargs
        gen_config = GenerationConfig(**generation_config)
        
        # Update model's generation config
        model.generation_config = gen_config
        
        print(f"  Updated Generation Config: {model.generation_config}")
        print("-" * 10 + "Starting Generation Process" + "-" * 10) 
    return model


def generate(rank, args):
    #visible_devices = [
    #    str(rank * args.gpus_per_model + i + args.begin_gpu) for i in range(args.gpus_per_model)
    #]
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

    
    # Set deterministic seeds - keep the original [42, 42] format in generation config
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
    tokenizer.padding_side = "left"

    # Ensure a pad_token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        # Ensure eos_token_id exists
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.pad_token_id
            tokenizer.eos_token = tokenizer.pad_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    prompt_lst = []
    with open(args.infile) as f:
        idx = 0
        for line in f.readlines():
            d = json.loads(line.strip())
            d["idx"] = idx
            prompt_lst.append(d)
            idx += 1

    print(f"the total number of prompts: {len(prompt_lst)}")
    prompt_lst = prompt_lst[rank :: args.num_processes]
    print(f"the total number of prompts for rank {rank}: {len(prompt_lst)}")
    
    if os.path.exists(args.outfile + f"{rank}"):
        generated = pd.read_json(args.outfile + f"{rank}", lines=True)
        remove_list = []
        for _ in range(len(prompt_lst)):
            if prompt_lst[_]["idx"] in generated["idx"].values and _ not in remove_list:
                remove_list.append(_)
        prompt_lst = [
            prompt_lst[_] for _ in range(len(prompt_lst)) if _ not in remove_list
        ]
    print(f"the total number of prompts for rank {rank} to generate: {len(prompt_lst)}")

    total_prompts = len(prompt_lst)
    s = time.time()
    max_new_tokens = args.max_new_tokens
    
    try:
        for start in tqdm(range(0, total_prompts, args.batch_size),
                         disable=rank != 0,
                         desc=f"Rank {rank} generation"):
            try:
                batch_start = time.time()
                
                if start % 20 == 0 and rank == 0:
                    print(f"rank {rank} has generated {start} prompts")
                    
                cur_prompt_lst = prompt_lst[start : start + args.batch_size]
                prompt_text = [f"{x['instructions']}" for x in cur_prompt_lst]
                model_inputs = tokenizer(
                    prompt_text, padding=True, add_special_tokens=True, return_tensors="pt"
                )
                input_ids = model_inputs["input_ids"].to(model.device)
                attention_mask = model_inputs["attention_mask"].to(model.device)
                prompt_len = input_ids.size(1)
                args.max_new_tokens = min(max_new_tokens, args.max_length - prompt_len)
                
                if args.max_new_tokens < 0:
                    generation_text = [""] * len(cur_prompt_lst)
                    for prompt, generation in zip(cur_prompt_lst, generation_text):
                        json_str = json.dumps(
                            {
                                "idx": prompt["idx"],
                                "completion": generation.strip(),
                            }
                        )
                        with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                            f.write(json_str + "\n")
                    continue

                # Split parameters into generation config and model inputs
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                generation_config = {
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "do_sample": False
                }

                decoding_configs = {
                    "topp": {"top_p": args.top_p, "do_sample": True, "seed": [42, 42]},
                    "topk": {"top_k": args.top_k, "do_sample": True, "seed": [42, 42]},
                    "greedy": {},
                    "temperature": {"temperature": args.temperature, "do_sample": True, "seed": [42, 42]},
                    "epsilon": {"epsilon_cutoff": args.epsilon_cutoff, "do_sample": True, "seed": [42, 42]},
                    "typical": {"typical_p": args.typical_p, "do_sample": True, "seed": [42, 42]},
                    "minp": {"min_p": args.min_p, "do_sample": True, "seed": [42, 42]},
                    "contrastive_search": {"penalty_alpha": args.cs_alpha, "top_k": args.cs_k},
                    "beam": {"num_beams": args.num_beams},
                    "diverse_beam": {
                        "num_beams": args.num_beams,
                        "num_beam_groups": args.num_beam_groups,
                        "diversity_penalty": 1.0
                    },
                    "assisted_decoding": {"prompt_lookup_num_tokens": args.prompt_lookup_num_tokens},
                    "dola": {"dola_layers": args.dola_layers, "repetition_penalty": 1.2}
                }

                if args.decoding_method not in decoding_configs:
                    raise ValueError(f"Unsupported decoding method: {args.decoding_method}")
                
                # Update generation config with method-specific parameters
                generation_config.update(decoding_configs[args.decoding_method])
                
                # Combine all parameters for generate()
                gen_kwargs = {**model_inputs, **generation_config}
                
                # Verify generation config (only for first batch)
                if start == 0:
                    model = verify_generation_config(model, generation_config, args.decoding_method, rank)
                
                outputs = model.generate(**gen_kwargs)

                generation_text = tokenizer.batch_decode(
                    outputs[:, prompt_len:],
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )
                
                for prompt, generation in zip(cur_prompt_lst, generation_text):
                    json_str = json.dumps(
                        {
                            "idx": prompt["idx"],
                            "completion": generation.strip(),
                        }
                    )
                    with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                        f.write(json_str + "\n")
                        
            except Exception as e:
                print(f"Error in batch processing at rank {rank}, start {start}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in generation process at rank {rank}: {e}")
        raise

    t = time.time()
    print(f"Time used for rank {rank}: {t - s:.2f}s")


if __name__ == "__main__":
    args = args_parse()
    args.early_stop = True
    #print(args)
    assert args.world_size % args.gpus_per_model == 0
    args.num_processes = args.world_size // args.gpus_per_model
    
    # Set deterministic seeds in the main process
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    
    if os.path.exists(args.outfile):
        try:
            all_ret = pd.read_json(args.outfile, lines=True)
            all_ret = all_ret.drop_duplicates(subset=["idx"], keep="first")
            all_ret.reset_index(drop=True, inplace=True)
            all_ret = all_ret[all_ret["completion"] != ""]
            all_ret.reset_index(drop=True, inplace=True)
            input_file = pd.read_json(args.infile, lines=True)
            if len(all_ret) == len(input_file):
                print(f"{args.outfile} already generated.")
                sys.exit(0)
            else:
                print("some prompts are not generated, regenerate them.")
                for _ in range(args.num_processes):
                    if os.path.exists(args.outfile + f"{_}"):
                        os.remove(args.outfile + f"{_}")
                for _ in range(len(all_ret)):
                    to_write_id = all_ret.iloc[_]["idx"] % args.num_processes
                    with open(
                        args.outfile + f"{to_write_id}", "a", encoding="utf-8"
                    ) as f:
                        json_str = json.dumps(
                            {
                                "idx": int(all_ret.iloc[_]["idx"]),
                                "completion": all_ret.iloc[_]["completion"],
                            }
                        )
                        f.write(json_str + "\n")
        except:
            print("bad output file")
            sys.exit(0)

    # Start processing
    process_list = []
    try:
        for i in range(args.num_processes):
            p = Process(target=generate, args=(i, args))
            p.start()
            process_list.append(p)
        
        for p in process_list:
            p.join()
            
        # Combine results
        all_ret = pd.DataFrame()
        for rank in range(args.num_processes):
            try:
                with open(args.outfile + f"{rank}", "r", encoding="utf-8") as f:
                    all_ret = pd.concat(
                        [all_ret, pd.read_json(f, lines=True)], ignore_index=True
                    )
            except Exception as e:
                print(f"Error reading temporary file {rank}: {e}")
                continue
                
        all_ret.sort_values(by="idx", inplace=True)
        all_ret.to_json(args.outfile, orient="records", lines=True, force_ascii=False)
        
        # Cleanup temporary files
        for rank in range(args.num_processes):
            try:
                os.remove(args.outfile + f"{rank}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {rank}: {e}")
                
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

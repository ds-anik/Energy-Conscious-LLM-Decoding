# Code for Open-Ended Generation Tasks

### Install Requirements
```
pip install datasets
pip install evaluate
pip install transformers
```

### Generation
Take Temperature For Example
```

python3 generate.py \
    --decoding_method temperature\
    --infile ./data_test/wikitext/Llama2_chat_input.jsonl\
    --outfile ./results/wikitext/output.jsonl\
    --model_name_or_path ${model_path}\
    --gpus_per_model 1\
    --world_size 1\
    --batch_size 1\
    --max_new_tokens 512\
    --temperature 0.7
```

## Acknowledgments

This code is based on [llm_decoding](https://github.com/DavidFanzz/llm_decoding?tab=readme-ov-file#code-for-a-thorough-examination-of-decoding-methods-in-the-era-of-llms) the official github repository for [A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/pdf/2402.06925) paper. Modifications have been made to adapt the code for our usecases. 
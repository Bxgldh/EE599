import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from trl import setup_chat_format


def load_llama(name,cache_dir):
    compute_dtype = getattr(torch, "float16")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        name,
        # device_map=device,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        local_files_only=True,   # 强制只用你刚缓存到本地的文件
        # use_auth_token=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        name, 
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=True,
        # use_auth_token=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer
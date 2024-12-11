import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig


def detect(use_cache = False, quantization_type = None):
    # load model
    torch.cuda.empty_cache()
    model_name = "./../gpt2"
    quantization_config = TorchAoConfig(quantization_type) if quantization_type else None
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config) if quantization_type else AutoModelForCausalLM.from_pretrained(model_name)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, quantization_config=quantization_config) if quantization_type else AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # inputs data
    input_text = "I like learning NLP because"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

    # detect
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    outputs = model.generate(input_ids, max_length=1024, use_cache=use_cache)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    end_time = time.time()
    num_tokens = outputs.shape[1] - input_ids.shape[1]

    delta_t = end_time - start_time
    speed = num_tokens/delta_t
    print(f"if use KV-cache: {use_cache}, quantization type: {quantization_type} tokens per second")
    print(f"time:{delta_t:.5f} seconds, speed: {speed:.5f}")
    print(f"peak GPU memory usage:{peak_memory:.5f} MB")



if __name__ == "__main__":
    detect(False)
    detect(True)
    detect(False, "int8_weight_only")
    detect(True, "int8_weight_only")
    detect(False, "int4_weight_only")
    detect(True, "int4_weight_only")
    detect(False, "int8_dynamic_activation_int8_weight")
    detect(True, "int8_dynamic_activation_int8_weight")
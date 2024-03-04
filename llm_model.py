!pip install -q torch datasets
!pip install -q accelerate==0.21.0 \
                peft==0.4.0 \
                bitsandbytes==0.40.2 \
                transformers==4.31.0 \
                trl==0.4.7
# !pip install transformers optimum accelerate peft trl auto-gptq bitsandbytes datasets==2.17.0
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
# from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from huggingface_hub import login
login(token="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")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from concurrent.futures import ThreadPoolExecutor

# Define global variables
input_tokens = 128
output_tokens = 128
total_tokens_per_batch = input_tokens + output_tokens
concurrency = 32

# Load the model and tokenizer
def optimize_model(model_name):
    # model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    return model

# Function to run model inference
def run_model_inference(model, tokenizer, prompt):
    start_time = time.time()
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_input, max_new_tokens=128)
    end_time = time.time()
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    inference_time = end_time - start_time
    print("Time Taken ::",inference_time)
    print("Response:: ",response)
    print("\n\n")
    # from here we can insert response into Database and return to frontend side

from concurrent.futures import ThreadPoolExecutor
import threading

def main():
    with ThreadPoolExecutor(max_workers=32) as executor:
        while True:
          try:
            model_path = input("Enter Hugging Face model path: ")
            model = optimize_model(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            question = input("Enter prompt: ")
            
            if question.lower().strip() == 'exit' or question.lower().strip() == 'quit':
                break

            print("Currently processing question:", question)
            print("Active thread count:", threading.active_count())

            if threading.active_count() < 32:
                print("While active thread:", threading.active_count())
                executor.submit(run_model_inference, model, tokenizer, question)
          except :
            print("Model not loaded")

if __name__ == "__main__":
    main()


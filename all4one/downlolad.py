# import os

# os.environ["HF_DATASETS_CACHE"] = "/home/Data/xac/nas/models"
# os.environ["HF_HOME"] = "/home/Data/xac/nas/models"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/Data/xac/nas/models"
# os.environ["TRANSFORMERS_CACHE"] = "/home/Data/xac/nas/models"

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_path="Phind/Phind-CodeLlama-34B-v2"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16
# )
# model.eval()

from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

# initialize the model

model_path = "Phind/Phind-CodeLlama-34B-v2"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
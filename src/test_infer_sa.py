
from transformers import AutoModelForCausalLM
from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
import torch
import tqdm

model_to_load = '../output/chess10k_gold/gpt2-model-bs1024-lr3e-4-ep40-20240820-104029'
tokenizer = CustomTokenizer.from_pretrained(model_to_load)
model = AutoModelForCausalLM.from_pretrained(model_to_load).to('cuda')
model = model.eval()

sa = ["rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRwKQkq-.0.1.."]

encoded_input = tokenizer(sa, return_tensors='pt')['input_ids']
sep = torch.full((len(sa),1), tokenizer.sep_token_id)
encoded_input = torch.cat([encoded_input, sep], dim=-1).to('cuda')

output = model.generate(encoded_input, max_new_tokens=1)
r = output[:, len(encoded_input[0]):]

print(tokenizer.batch_decode(r.numpy()))
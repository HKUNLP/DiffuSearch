
from transformers import AutoModelForCausalLM, AutoConfig
from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
from llmtuner.tuner.ddm.model import DiffusionModel
import torch
import torch.nn.functional as F
import os
import numpy as np
import tqdm


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def topk_decoding(
        x0, 
        x0_scores,
        decoding_strategy,
        init_maskable_mask, 
        t,
        max_step,
        noise
    ):
        # decoding_strategy needs to take the form of "<topk_mode>-<schedule>"
        topk_mode, schedule = decoding_strategy.split("-")

        # select rate% not confident tokens, ~1 -> 0
        if schedule == "linear":
            rate = t / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError
        
        # compute the cutoff length for denoising top-k positions
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
        # set the scores of unmaskable symbols to a large value so that they will never be selected
        _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError

        ### recovered tokens can also be remasked based on current scores
        masked_to_noise = lowest_k_mask
        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt

def generate_samples(inputs, verbose=True):
    """
        select 1/T% tokens to denoise at each step
    """
    model.cuda()
    model.eval()

    # x = torch.transpose(torch.stack(inputs['input_ids']), 0, 1).cuda()
    # src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()
    x = inputs['input_ids'].cuda()
    src_mask = inputs['src_mask'].bool().cuda()
    attention_mask = torch.ones_like(x) 
    batch_size = x.size(0)

    init_maskable_mask = maskable_mask = ~src_mask
    next_action_position = src_mask.int().argmin(dim=-1, keepdim=True)

    for t in range(diffusion_steps-1, -1, -1): # t from T-1 to 0
        with torch.no_grad():
            if t == diffusion_steps-1:
                # first forward, all position except src is [M]
                xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

            if verbose:
                print(f"t={t+1}(in):", tokenizer.decode(xt.tolist()[0]))

            t_tensor = torch.full((batch_size, ), t, device=x.device)
            logits = model(xt, t_tensor, attention_mask=attention_mask)
            scores = torch.log_softmax(logits, dim=-1)
            scores[:,:,tokenizer.vocab_size:]=-1000
            x0_scores, x0 = scores.max(-1)

            #### deal with shift, left most token will be replaced anyway
            x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
            x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
            
            #### replace output of non-[MASK] positions with xt
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
            if verbose:
                print(f"t={t+1}(out):", tokenizer.decode(x0.tolist()[0]))
            
            if t > 0:
                # maskable_mask indicates all mask positions currently
                # _reparam_decoding redecides mask tokens in xt based on scores
                xt = topk_decoding(
                    x0,
                    x0_scores,
                    "stochastic0.5-linear",
                    init_maskable_mask, 
                    t,
                    diffusion_steps,
                    tokenizer.mask_token_id
                )
                xt[:, next_action_position] =  tokenizer.mask_token_id
            else:
                xt = x0
    return xt


max_length = 328
model_to_load = '../output/chess10k_gold_s_asa/ddm-tiny-bs1024-lr3e-4-ep200-T20-20240828-094752'

diffusion_steps = 10
tokenizer = CustomTokenizer.from_pretrained(model_to_load)
config = AutoConfig.from_pretrained(model_to_load)
model = AutoModelForCausalLM.from_config(config)
model = DiffusionModel(model, config, None)

load_path = os.path.join(model_to_load, 'pytorch_model.bin')
loaded = torch.load(load_path,map_location=torch.device('cuda'))
model.load_state_dict(loaded, strict=False)
model = model.eval()

# src = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRwKQkq-.0.1.."
src = 'rn.qkb.rppp.pppp.....n.....p.bB....P......N..N..PPP.PPPPR..QKB.RbKQkq-.5.4..'
tgt = "a1b1"  ## a random move, just as a place holder

src_ids = tokenizer.encode(src)
tgt_ids = tokenizer.encode(tgt)
input_ids = src_ids + [tokenizer.sep_token_id] + tgt_ids + [tokenizer.eos_token_id]
input_ids = torch.tensor(input_ids)[None]
pad = torch.full((1, max_length-len(input_ids[0])), tokenizer.pad_token_id)
encoded_input = torch.cat([input_ids, pad], dim=-1)
src_mask = torch.tensor([1] * (len(src_ids) + 1) + [0]*(max_length-1-len(src_ids)))[None]
# print(encoded_input.shape)
pack_input = {"input_ids": encoded_input, "src_mask": src_mask}

with torch.no_grad():
    x0 = generate_samples(pack_input, verbose=False)
    res = tokenizer.batch_decode(x0.cpu().numpy())[0]
    print(res)
    print("next move is:", res.split('[SEP]')[1].split(' ')[0])
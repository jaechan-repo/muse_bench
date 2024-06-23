import sys
sys.path.append(".")
sys.path.append("../baselines")

from baselines.utils import load_model_and_tokenizer
from baselines.dataset import DefaultDataset

import os
from typing import List, Dict
import torch
from tqdm import tqdm
import zlib
import numpy as np


def compute_ppl(text: str, model, tokenizer, device='cuda'):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ppl = torch.exp(loss).item()
    return ppl, all_prob, loss.item()


def inference(text: str, model, ref_model, tokenizer, ref_tokenizer) -> Dict:
    pred = {}

    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
    _, _, p_ref_likelihood = compute_ppl(text, ref_model, ref_tokenizer, device=ref_model.device)
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

    pred["PPL"] = float(p1_likelihood)
    pred["PPL/ref"] = float(p1_likelihood / p_ref_likelihood)
    pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
    pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

    return pred


def evaluate_data(test_data: Dict,
                  model, ref_model,
                  tokenizer, ref_tokenizer,
                  col_name: str = 'text'):
    out = []
    for row in tqdm(test_data):
        text = row[col_name]
        row['ppl'] = inference(text, model, ref_model, tokenizer, ref_tokenizer)
        out.append(row)
    return out


def main(data_file: str, 
         model_dir: str,
         model_name: str,
         tokenizer_dir: str,
         ref_model_dir: str,
         reinforced_model_dir: str | None = None):
    model, tokenizer = load_model_and_tokenizer(model_dir,
                                                tokenizer_dir=tokenizer_dir,
                                                reinforced_model_dir=reinforced_model_dir,
                                                model_name=model_name)
    ref_model, ref_tokenizer = load_model_and_tokenizer(ref_model_dir,
                                                        tokenizer_dir=tokenizer_dir)

    assert os.path.splitext(data_file)[-1] in {'.txt', '.json'}

    dataset = DefaultDataset(
        file_path=data_file,
        tokenizer=tokenizer,
        add_bos_token=True
    )
    data = [{'data_file': data_file,
             'text': text} for text in dataset.strings]

    out: List[Dict] = evaluate_data(
        data,
        model=model, ref_model=ref_model,
        tokenizer=tokenizer, ref_tokenizer=ref_tokenizer,
        col_name='text'
    )
    return out

import sys
sys.path.append(".")
sys.path.append("../baselines")
sys.path.append("../verbmem")

from baselines.utils import output_text, output_json, load_tokenizer
from baselines.dataset import DefaultDataset

import argparse
from datasets import load_dataset
import os
import json
from typing import List, Dict
import random


ROOT_DIR = '.'
TEMP_DIR = './temp'
N_FOLDS = 4
HF_DATASET_DIRS = {
    'bbc': "chan030609/MUSE-News-2",
    'hp': "chan030609/MUSE-Books-2"
}


def output_tokenization(in_file: str, out_file: str, tokenizer,
                        max_len: int = 2048, add_bos_token: bool = True):
    assert in_file.endswith('.txt') and out_file.endswith('.json')
    dataset = DefaultDataset(file_path=in_file,
                             tokenizer=tokenizer,
                             max_len=max_len,
                             add_bos_token=add_bos_token)
    data = [{
        'text': text,
        'input_ids': input_ids.tolist()
    } for text, input_ids in zip(dataset.strings, dataset.input_ids)]
    output_json(data, out_file)
    return data


def main(args):

    for corpus in ['bbc', 'hp']:
        # 1. Download original corpus and tokenize
        tokenizer = load_tokenizer(args.tokenizer_dir, add_pad_token=True, use_fast=True)

        for split in ['forget', 'holdout', 'retain1', 'retain2']:
            ds: List[str] = load_dataset(HF_DATASET_DIRS[corpus], 'main', split=split)['text']
            output_json(ds, os.path.join(ROOT_DIR, f"{corpus}/main/{split}_raw.json"))
            output_text('\n\n'.join(ds), os.path.join(ROOT_DIR, f"{corpus}/main/{split}.txt"))
            output_tokenization(os.path.join(ROOT_DIR, f"{corpus}/main/{split}.txt"),
                                os.path.join(ROOT_DIR, f"{corpus}/main/{split}.json"),
                                tokenizer=tokenizer, add_bos_token=not args.skip_bos_token, max_len=args.max_len)

        if corpus == 'bbc':
            for k in range(1, N_FOLDS + 1):
                for criterion in ['sust', 'scal']:
                    ds: List[str] = load_dataset(HF_DATASET_DIRS[corpus], criterion, split=f'forget_{k}')['text']
                    output_json(ds, os.path.join(ROOT_DIR, f"{corpus}/{criterion}/forget_{k}_raw.json"))
                    output_text('\n\n'.join(ds), os.path.join(ROOT_DIR, f"{corpus}/{criterion}/forget_{k}.txt"))
                    output_tokenization(os.path.join(ROOT_DIR, f"{corpus}/{criterion}/forget_{k}.txt"),
                                        os.path.join(ROOT_DIR, f"{corpus}/{criterion}/forget_{k}.json"),
                                        tokenizer=tokenizer, add_bos_token=not args.skip_bos_token, max_len=args.max_len)

        # 2. Sample from the corpus for PrivLeak and VerbMem
        for split in ['forget', 'retain2', 'holdout']:
            with open(os.path.join(ROOT_DIR, f"{corpus}/main/{split}.json"), 'r') as f:
                ds = json.load(f)
            random.seed(args.seed)
            subds = random.sample(ds, args.privleak_sample_k)
            output_json(subds, os.path.join(ROOT_DIR, f"{corpus}/privleak/{split}.json"))

        with open(os.path.join(ROOT_DIR, f"{corpus}/main/forget.json"), 'r') as f:
            ds = json.load(f)
        random.seed(args.seed)
        subds = random.sample(ds, args.verbmem_sample_k)
        output_json(subds, os.path.join(ROOT_DIR, f"{corpus}/verbmem/forget.json"))

        # 3-1. Download QA pairs for KnowMem
        for split in ['forget', 'retain2']:
            ds: List[Dict] = load_dataset(HF_DATASET_DIRS[corpus], 'knowmem', split=f"{split}_qa")
            ds = [{'question': q, 'answer': a} for q, a in zip(ds['question'], ds['answer'])]
            output_json(ds, os.path.join(ROOT_DIR, f"{corpus}/knowmem/{split}_qa.json"))

        # 3-2. Download QA in-context learning shots
        for split in ['forget', 'retain2']:
            ds: List[Dict] = load_dataset(HF_DATASET_DIRS[corpus], 'knowmem', split=f"{split}_qa_icl")
            ds = [{'question': q, 'answer': a} for q, a in zip(ds['question'], ds['answer'])]
            output_json(ds, os.path.join(ROOT_DIR, f"{corpus}/knowmem/{split}_qa_icl.json"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_dir', type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Directory to the tokenizer. Has NO effect if both the VerbMem and PrivLeak splits are downloaded from HF, instead of constructed with this tokenizer.")
    parser.add_argument('--max_len', type=int, default=2048,
                        help="Maximum length of input to the model, measured in the number of tokens.")
    parser.add_argument('--skip_bos_token', action='store_true',
                        help="Defaults to false.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for sampling sequences from the corpus. This seed is set every time a random sampling occurs.")
    parser.add_argument('--privleak_sample_k', type=int, default=100,
                        help="How many sequences to sample from the raw corpus (`forget`, `retain2`, and `holdout` sets) for PrivLeak evaluation. Has NO effect if `download_privleak` is enabled.")
    parser.add_argument('--verbmem_sample_k', type=int, default=100,
                        help="How many sequences to sample from the raw corpus (`forget` set) for VerbMem evaluation. Has NO effect if `download_verbmem` is enabled.")
    args = parser.parse_args()
    main(args)

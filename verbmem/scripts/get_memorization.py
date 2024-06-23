import sys
sys.path.append(".")
sys.path.append("../baselines")

from get_rouge import main as get_rouge
from baselines.utils import output_json, file_exists, read_json

import os
from typing import Literal, List
import argparse


DATA_FILES = {
    corpus: f"../data/{corpus}/verbmem/forget.json"
    for corpus in ['bbc', 'hp']
}
STEP_ALGOS = ['ga', 'ga_gdr', 'ga_klr', 'npo', 'npo_gdr', 'npo_klr']
ALPHA_ALGOS = ['tv', 'whp']
CKPT_NAMES = {
    'bbc': ['ga/step=102',
            'ga_gdr/step=714',
            'ga_klr/step=1020',
            'npo/step=102',
            'npo_gdr/step=1020',
            'npo_klr/step=1020',
            'tv/alpha=16',
            'whp/alpha=4'],
    'hp': []
}
TARGET_MODEL_DIRS = {
    'bbc': "swj0419/bbc-original_STEP0000080_5-31",
    'hp': "swj0419/hpall_STEP0000160-5-13"
}
RETRAIN_MODEL_DIRS = {
    'bbc': "swj0419/bbc-retrain_STEP0000040_5-31",
    'hp': "swj0419/hpretrain_STEP0000080-5-5"
}
CKPT_DIR = "../baselines/ckpt"
REINFORCED_MODEL_NAME = "tv_ft"
TOKENIZER_DIR = "meta-llama/Llama-2-7b-hf"


def parse_slash(input: str):
    def parse_equal(input: str):
        if '=' not in input:
            raise ValueError("Equal sign not present.")
        i = input.rfind('=')
        return input[i + 1:]

    if '/' not in input:
        return input, None
    i = input.rfind('/')
    prefix = input[:i]
    suffix = input[i + 1:]
    return prefix, parse_equal(suffix)


def main(corpus: Literal['bbc', 'hp'],
         ckpt_names: List[str],
         data_file: str,
         target_model_dir: str,
         retrain_model_dir: str,
         reinforced_model_dir: str,
         max_new_tokens: int = 128,
         tokenizer_dir: str = TOKENIZER_DIR,
         step_algos: List[str] = STEP_ALGOS,
         alpha_algos: List[str] = ALPHA_ALGOS,
         ckpt_dir: str = "../baselines/ckpt"):
    kwargs = {
        'data_file': data_file,
        'tokenizer_dir': tokenizer_dir,
        'max_new_tokens': max_new_tokens,
    }

    # Target model
    model_name = 'target'
    print(f"Evaluating {model_name}...")
    out_dir = f"./out/{corpus}/{model_name}/forget"
    if file_exists(out_dir):
        print(f"File already exists for {model_name} at {out_dir}; continuing.")
    else:
        out, history = get_rouge(model_dir=target_model_dir, model_name=model_name, **kwargs)
        output_json(out, f"{out_dir}/out.json")
        output_json(history, f"{out_dir}/history.json")

    # Retrained model
    model_name = 'retrain'
    print(f"Evaluating {model_name}...")
    out_dir = f"./out/{corpus}/{model_name}/forget"
    if file_exists(out_dir):
        print(f"File already exists for {model_name} at {out_dir}; continuing.")
    else:
        out, history = get_rouge(model_dir=retrain_model_dir, model_name=model_name, **kwargs)
        output_json(out, f"{out_dir}/out.json")
        output_json(history, f"{out_dir}/history.json")

    # Unlearned models
    for model_name in ckpt_names:
        print(f"Evaluating {model_name}...")
        out_dir = f"./out/{corpus}/{model_name}/forget"
        if file_exists(out_dir):
            print(f"File already exists at {out_dir}; continuing.")
            continue;
        algo, ckpt = parse_slash(model_name)  # Expected format: "ga/5"
        if algo in step_algos:
            model_dir = f"{ckpt_dir}/{corpus}/{algo}" + (
                f"/checkpoint-{int(ckpt)}"
                if ckpt is not None
                else ""
            )
            curr_reinforced_model_dir = None
        elif algo in alpha_algos:
            model_dir = target_model_dir
            curr_reinforced_model_dir = reinforced_model_dir
        else: raise ValueError("Unrecognized algo name.")
        out, history = get_rouge(model_dir=model_dir,
                                    model_name=model_name,
                                    reinforced_model_dir=curr_reinforced_model_dir,
                                    **kwargs)
        output_json(out, f"{out_dir}/out.json")
        output_json(history, f"{out_dir}/history.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus', type=str, required=True, choices=['bbc', 'hp'],
                        help="Corpus to evaluate against. Either BBC or HP.")

    parser.add_argument('--ckpt_names', type=str, nargs='+', default=[])
    parser.add_argument('--step_algos', type=str, nargs='+', default=STEP_ALGOS)
    parser.add_argument('--alpha_algos', type=str, nargs='+', default=ALPHA_ALGOS)
    parser.add_argument('--ckpt_dir', type=str, default=CKPT_DIR)

    parser.add_argument('--max_new_tokens', type=int, default=128)

    parser.add_argument('--data_file', type=str, default=None)

    parser.add_argument('--target_model_dir', type=str, default=None)
    parser.add_argument('--retrain_model_dir', type=str, default=None)
    parser.add_argument('--reinforced_model_dir', type=str, default=None)

    parser.add_argument('--tokenizer_dir', type=str, default=TOKENIZER_DIR)

    args = parser.parse_args()

    if not args.ckpt_names:
        args.ckpt_names = CKPT_NAMES[args.corpus]
    if not args.target_model_dir:
        args.target_model_dir = TARGET_MODEL_DIRS[args.corpus]
    if not args.retrain_model_dir:
        args.retrain_model_dir = RETRAIN_MODEL_DIRS[args.corpus]
    if not args.reinforced_model_dir:
        args.reinforced_model_dir = f"{args.ckpt_dir}/{args.corpus}/{REINFORCED_MODEL_NAME}"

    if not args.data_file:
        args.data_file = DATA_FILES[args.corpus]
    main(**vars(args))

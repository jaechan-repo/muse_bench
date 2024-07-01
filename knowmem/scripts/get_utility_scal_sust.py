import sys
sys.path.append(".")
sys.path.append("../baselines")

from get_rouge_qa import main as get_rouge_qa
from baselines.utils import output_json, file_exists, read_json

import os
from typing import Literal, List
import argparse


DATA_FILES = {
    corpus: {
        'qa_file': f"../data/{corpus}/knowmem/retain2_qa.json",
        'icl_file': f"../data/{corpus}/knowmem/retain2_qa_icl.json"
    }
    for corpus in ['bbc', 'hp']
}
STEP_ALGOS = ['ga', 'ga_gdr', 'ga_klr', 'npo', 'npo_gdr', 'npo_klr']
ALPHA_ALGOS = ['tv', 'whp']
CKPT_NAMES = {
    'bbc': ['ga',
            'ga_gdr',
            'ga_klr',
            'npo',
            'npo_gdr',
            'npo_klr'],
    'hp': ['ga/step=139',
           'ga_gdr/step=139',
           'ga_klr/step=556',
           'npo/step=139',
           'npo_gdr/step=139',
           'npo_klr/step=417',
           'tv/alpha=1',
           'whp/alpha=256']}
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
         folds: List[int],
         mode: str,
         qa_file: str,
         icl_file: str,
         max_new_tokens: int = 32,
         tokenizer_dir: str = TOKENIZER_DIR,
         step_algos: List[str] = STEP_ALGOS,
         alpha_algos: List[str] = ALPHA_ALGOS,
         ckpt_dir: str = "../baselines/ckpt"):
    kwargs = {
        'qa_file': qa_file,
        'icl_file': icl_file,
        'tokenizer_dir': tokenizer_dir,
        'max_new_tokens': max_new_tokens,
    }

    # Unlearned models
    for model_name in ckpt_names:
        for fold in folds:
            print(f"Evaluating {model_name} at fold {fold}...")
            out_dir = f"./out/{corpus}/{mode}/{model_name}/{fold}/retain"
            if file_exists(out_dir):
                print(f"File already exist at {out_dir}; continuing.")
                continue;
            algo, ckpt = parse_slash(model_name)  # Expected format: "ga/5"
            if algo in step_algos:
                model_dir = f"{ckpt_dir}/{corpus}/{mode}/{algo}/fold={fold}" + (
                    f"/checkpoint-{int(ckpt)}"
                    if ckpt is not None
                    else ""
                )
                curr_reinforced_model_dir = None
            elif algo in alpha_algos: raise NotImplementedError()
            else: raise ValueError("Unrecognized algo name.")
            out, history = get_rouge_qa(model_dir=model_dir,
                                        model_name=model_name,
                                        reinforced_model_dir=curr_reinforced_model_dir,
                                        **kwargs)
            output_json(out, f"{out_dir}/out.json")
            output_json(history, f"{out_dir}/history.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus', type=str, required=True, choices=['bbc', 'hp'],
                        help=".")
    parser.add_argument('--mode', type=str, required=True, choices=['scal', 'sust'],
                        help=".")
    parser.add_argument('--folds', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--ckpt_names', type=str, nargs='+', default=[])
    parser.add_argument('--step_algos', type=str, nargs='+', default=STEP_ALGOS)
    parser.add_argument('--alpha_algos', type=str, nargs='+', default=ALPHA_ALGOS)
    parser.add_argument('--ckpt_dir', type=str, default=CKPT_DIR)

    parser.add_argument('--max_new_tokens', type=int, default=32)

    parser.add_argument('--qa_file', type=str, default=None)
    parser.add_argument('--icl_file', type=str, default=None)

    # parser.add_argument('--target_model_dir', type=str, default=None)
    # parser.add_argument('--reinforced_model_dir', type=str, default=None)

    parser.add_argument('--tokenizer_dir', type=str, default=TOKENIZER_DIR)

    args = parser.parse_args()

    if not args.ckpt_names:
        args.ckpt_names = CKPT_NAMES[args.corpus]
    # if not args.target_model_dir:
    #     args.target_model_dir = TARGET_MODEL_DIRS[args.corpus]
    # if not args.retrain_model_dir:
    #     args.retrain_model_dir = RETRAIN_MODEL_DIRS[args.corpus]
    # if not args.reinforced_model_dir:
    #     args.reinforced_model_dir = f"{args.ckpt_dir}/{args.corpus}/{REINFORCED_MODEL_NAME}"

    if not args.qa_file:
        args.qa_file = DATA_FILES[args.corpus]['qa_file']
    if not args.icl_file:
        args.icl_file = DATA_FILES[args.corpus]['icl_file']
    main(**vars(args))

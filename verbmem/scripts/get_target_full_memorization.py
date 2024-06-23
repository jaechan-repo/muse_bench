import sys
sys.path.append(".")
sys.path.append("../baselines")

from get_rouge import main as get_rouge
from baselines.utils import output_json, file_exists

from typing import Literal
import argparse


DATA_FILES = {
    corpus: f"../data/{corpus}/verbmem/forget_full.json"
    for corpus in ['bbc', 'hp']
}
TARGET_MODEL_DIRS = {
    'bbc': "swj0419/bbc-original_STEP0000080_5-31",
    'hp': "swj0419/hpall_STEP0000160-5-13"
}
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
         data_file: str,
         target_model_dir: str,
         max_new_tokens: int = 32,
         tokenizer_dir: str = TOKENIZER_DIR,):
    kwargs = {
        'data_file': data_file,
        'tokenizer_dir': tokenizer_dir,
        'max_new_tokens': max_new_tokens,
    }
    model_name = 'target'
    print(f"Evaluating {model_name}...")
    out_dir = f"./out/{corpus}/{model_name}/forget"
    if file_exists(out_dir):
        print(f"File already exists for {model_name} at {out_dir}; continuing.")
    else:
        out, history = get_rouge(model_dir=target_model_dir, model_name=model_name, **kwargs)
        output_json(out, f"{out_dir}/out.json")
        output_json(history, f"{out_dir}/history.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus', type=str, required=True, choices=['bbc', 'hp'],
                        help="Corpus to evaluate against. Either BBC or HP.")
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--target_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer_dir', type=str, default=TOKENIZER_DIR)

    args = parser.parse_args()

    if not args.target_model_dir:
        args.target_model_dir = TARGET_MODEL_DIRS[args.corpus]
    if not args.data_file:
        args.data_file = DATA_FILES[args.corpus]
    main(**vars(args))

import sys
sys.path.append(".")
sys.path.append("../baselines")

from get_rouge_qa import main as get_rouge_qa
from baselines.utils import output_json, file_exists, read_json

import argparse
from typing import List


DATA_FILES = {
    corpus: {
        'qa_file': f"../data/{corpus}/knowmem/forget_qa.json",
        'icl_file': f"../data/{corpus}/knowmem/forget_qa_icl.json"
    }
    for corpus in ['bbc', 'hp']
}
STEP_ALGOS = ['ga', 'ga_gdr', 'ga_klr', 'npo', 'npo_gdr', 'npo_klr']
ALPHA_ALGOS = ['tv', 'whp']
TARGET_MODEL_DIRS = {
    'bbc': "swj0419/bbc-original_STEP0000080_5-31",
    'hp': "swj0419/hpall_STEP0000160-5-13"
}
RETRAIN_MODEL_DIRS = {
    'bbc': "swj0419/bbc-retrain_STEP0000040_5-31",
    'hp': "swj0419/hpretrain_STEP0000080-5-5"
}
STEPS = {
    'bbc': [102 * i for i in range(1, 11)],
    'hp': [139 * i for i in range(1, 11)]
}
ALPHAS = {
    'bbc': [2 ** i for i in range(10)], 
    'hp': [2 ** i for i in range(10)]
}

CKPT_DIR = "../baselines/ckpt"
REINFORCED_MODEL_NAME = "tv_ft"
TOKENIZER_DIR = "meta-llama/Llama-2-7b-hf"


def main(corpus: str,
         algos: List[str],
         steps: List[int],
         alphas: List[float],
         qa_file: str,
         icl_file: str,
         target_model_dir: str,
         retrain_model_dir: str,
         reinforced_model_dir: str,
         max_new_tokens: int = 32,
         metric: str = 'mean_rougeL',
         tokenizer_dir: str = TOKENIZER_DIR,
         step_algos: List[str] = STEP_ALGOS,
         alpha_algos: List[str] = ALPHA_ALGOS,
         ckpt_dir: str = "../baselines/ckpt",):
    kwargs = {
        'qa_file': qa_file,
        'icl_file': icl_file,
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
        out, history = get_rouge_qa(model_dir=target_model_dir, model_name=model_name, **kwargs)
        output_json(out, f"{out_dir}/out.json")
        output_json(history, f"{out_dir}/history.json")

    # Retrained model
    model_name = 'retrain'
    print(f"Evaluating {model_name}...")
    out_dir = f"./out/{corpus}/{model_name}/forget"
    if file_exists(out_dir):
        print(f"File already exists at {out_dir}; loading its data.")
        out = read_json(f"{out_dir}/out.json")
    else:
        out, history = get_rouge_qa(model_dir=retrain_model_dir, model_name=model_name, **kwargs)
        output_json(out, f"{out_dir}/out.json")
        output_json(history, f"{out_dir}/history.json")

    score_retrain: float = out[metric]

    # Unlearned models
    for algo in algos:
        if algo in step_algos:
            for step in steps:
                model_name = f"{algo}/step={step}"
                print(f"Evaluating {model_name}...")
                out_dir = f"./out/{corpus}/{model_name}/forget"
                if file_exists(out_dir):
                    print(f"File already exists at {out_dir}; continuing.")
                    out = read_json(f"{out_dir}/out.json")
                    if out[metric] <= score_retrain: break
                    else: continue
                model_dir = f"{ckpt_dir}/{corpus}/{algo}/checkpoint-{step}"
                out, history = get_rouge_qa(model_dir=model_dir, model_name=model_name, **kwargs)
                output_json(out | {f"{metric}_retrain": score_retrain}, f"{out_dir}/out.json")
                output_json(history, f"{out_dir}/history.json")
                if out[metric] <= score_retrain: break

        if algo in alpha_algos:
            for alpha in alphas:
                model_name =  f"{algo}/alpha={alpha}"
                print(f"Evaluating {model_name}...")
                out_dir = f"./out/{corpus}/{model_name}/forget"
                if file_exists(out_dir):
                    print(f"File already exists at {out_dir}; continuing.")
                    out = read_json(f"{out_dir}/out.json")
                    if out[metric] <= score_retrain: break
                    else: continue
                out, history = get_rouge_qa(model_dir=target_model_dir,
                                            model_name=model_name,
                                            reinforced_model_dir=reinforced_model_dir,
                                            **kwargs)
                output_json(out | {f"{metric}_retrain": score_retrain}, f"{out_dir}/out.json")
                output_json(history, f"{out_dir}/history.json")
                if out[metric] <= score_retrain: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, choices=['bbc', 'hp'],
                        help="Corpus to evaluate against. Either BBC or HP.")
    parser.add_argument('--algos', type=str, nargs='+',
                        default = STEP_ALGOS + ALPHA_ALGOS)
    parser.add_argument('--metric', type=str, default='mean_rougeL')
    parser.add_argument('--max_new_tokens', type=int, default=32)

    parser.add_argument('--qa_file', type=str, default=None)
    parser.add_argument('--icl_file', type=str, default=None)

    parser.add_argument('--target_model_dir', type=str, default=None)
    parser.add_argument('--retrain_model_dir', type=str, default=None)
    parser.add_argument('--reinforced_model_dir', type=str, default=None)

    parser.add_argument('--step_algos', type=str, nargs='+', default=STEP_ALGOS)
    parser.add_argument('--alpha_algos', type=str, nargs='+', default=ALPHA_ALGOS)

    parser.add_argument('--steps', type=int, nargs='+', default=None)
    parser.add_argument('--alphas', type=float, nargs='+', default=None)

    parser.add_argument('--ckpt_dir', type=str, default=CKPT_DIR)
    parser.add_argument('--tokenizer_dir', type=str, default=TOKENIZER_DIR)

    args = parser.parse_args()

    if not args.target_model_dir:
        args.target_model_dir = TARGET_MODEL_DIRS[args.corpus]
    if not args.retrain_model_dir:
        args.retrain_model_dir = RETRAIN_MODEL_DIRS[args.corpus]
    if not args.reinforced_model_dir:
        args.reinforced_model_dir = f"{args.ckpt_dir}/{args.corpus}/{REINFORCED_MODEL_NAME}"

    if not args.steps:
        args.steps = STEPS[args.corpus]
    if not args.alphas:
        args.alphas = ALPHAS[args.corpus]

    if not args.qa_file:
        args.qa_file = DATA_FILES[args.corpus]['qa_file']
    if not args.icl_file:
        args.icl_file = DATA_FILES[args.corpus]['icl_file']

    main(**vars(args))

import sys
sys.path.append(".")
sys.path.append("../baselines")

from get_ppl_utils import main as get_ppl
from baselines.utils import output_json, read_json

import os
from typing import Literal, List
import argparse
import numpy as np
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve


DATA_FILES = {
    'bbc': {
        'forget': "../data/bbc/privleak/forget.json",
        'retain2': "../data/bbc/privleak/retain2.json",
        'holdout': "../data/bbc/privleak/holdout.json"
    },
    'hp': {
        'forget': "../data/hp/privleak/forget.json",
        'retain2': "../data/hp/privleak/retain2.json",
        'holdout': "../data/hp/privleak/holdout.json"
    }
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
            'tv/alpha=512',
            'whp/alpha=4'],
    'hp': ['ga/step=139',
           'ga_gdr/step=139',
           'ga_klr/step=556',
           'npo/step=139',
           'npo_gdr/step=139',
           'npo_klr/step=417',
           'tv/alpha=512',
           'whp/alpha=256']
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


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc


def main(corpus: Literal['bbc', 'hp'],
         ckpt_names: List[str],
         forget_file: str,
         retain_file: str,
         holdout_file: str,
         tokenizer_dir: str,
         target_model_dir: str,
         retrain_model_dir: str,
         step_algos: List[str],
         alpha_algos: List[str],
         ckpt_dir: str = "../baselines/ckpt",
         reinforced_model_name: str = "tv_ft"):
    for split, data_file in zip(['forget', 'retain2', 'holdout'],
                                [forget_file, retain_file, holdout_file]):
        kwargs = {
            'data_file': data_file,
            'tokenizer_dir': tokenizer_dir,
            'ref_model_dir': tokenizer_dir
        }

        # Target model
        model_name = 'target'
        print(f"Evaluating {model_name} on {split} set...")
        out_file = f"./out/{corpus}/{model_name}/{split}.json"
        if os.path.isfile(out_file): print(f"File already exists for {model_name} at {out_file}; continuing.")
        else:
            out = get_ppl(model_dir=target_model_dir, model_name=model_name, **kwargs)
            output_json(out, out_file)

        # Retrain model
        model_name = 'retrain'
        print(f"Evaluating {model_name} on {split} set...")
        out_file = f"./out/{corpus}/{model_name}/{split}.json"
        if os.path.isfile(out_file): print(f"File already exists for {model_name} at {out_file}; continuing.")
        else:
            out = get_ppl(model_dir=retrain_model_dir, model_name=model_name, **kwargs)
            output_json(out, out_file)
            
        # Unlearned models
        for model_name in ckpt_names:
            print(f"Evaluating {model_name} on {split} set...")
            out_file = f"./out/{corpus}/{model_name}/{split}.json"
            if os.path.isfile(out_file):
                print(f"File already exists for {model_name} at {out_file}; continuing.")
                continue
            algo, ckpt = parse_slash(model_name)  # Expected format: "ga/5"
            if algo in step_algos:
                model_dir = f"{ckpt_dir}/{corpus}/{algo}" + (
                    f"/checkpoint-{int(ckpt)}"
                    if ckpt is not None
                    else ""
                )
                reinforced_model_dir = None
            elif algo in alpha_algos:
                model_dir = target_model_dir
                reinforced_model_dir = f"{ckpt_dir}/{corpus}/{reinforced_model_name}"
            else: raise ValueError("Unrecognized algo name.")
            out = get_ppl(model_dir=model_dir,
                          model_name=model_name,
                          reinforced_model_dir=reinforced_model_dir,
                          **kwargs)
            output_json(out, out_file)

    for model_name in ckpt_names + ['target', 'retrain']:
        auc_file = f"./out/{corpus}/{model_name}/auc.json"
        aucs = {}
        for split0 in ['forget', 'retain2', 'holdout']:
            for split1 in ['forget', 'retain2', 'holdout']:
                out0 = read_json(f"./out/{corpus}/{model_name}/{split0}.json")
                out1 = read_json(f"./out/{corpus}/{model_name}/{split1}.json")
                res = {}
                for ppl_key in out0[0]['ppl']:
                    ppl_nonmember = [d['ppl'][ppl_key] for d in out0]
                    ppl_member = [d['ppl'][ppl_key] for d in out1]
                    ppl = np.array(ppl_nonmember + ppl_member)
                    y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                    _, _, auc, _ = sweep(ppl, y)
                    res[ppl_key] = auc
                aucs[f"{split0}_{split1}"] = res
        output_json(aucs, auc_file)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus', type=str, required=True, choices=['bbc', 'hp'],
                        help="Corpus to evaluate against. Either BBC or HP.")

    parser.add_argument('--ckpt_names', type=str, nargs='+', default=[])
    parser.add_argument('--ckpt_dir', type=str, default=CKPT_DIR)
    parser.add_argument('--tokenizer_dir', type=str, default=TOKENIZER_DIR)
    parser.add_argument('--target_model_dir', type=str, default=None)
    parser.add_argument('--retrain_model_dir', type=str, default=None)
    parser.add_argument('--reinforced_model_name', type=str, default=REINFORCED_MODEL_NAME)
    parser.add_argument('--forget_file', type=str, default=None)
    parser.add_argument('--retain_file', type=str, default=None)
    parser.add_argument('--holdout_file', type=str, default=None)
    parser.add_argument('--step_algos', type=str, nargs='+', default=STEP_ALGOS)
    parser.add_argument('--alpha_algos', type=str, nargs='+', default=ALPHA_ALGOS)
    args = parser.parse_args()

    if not args.ckpt_names:
        args.ckpt_names = CKPT_NAMES[args.corpus]
    if not args.target_model_dir:
        args.target_model_dir = TARGET_MODEL_DIRS[args.corpus]
    if not args.retrain_model_dir:
        args.retrain_model_dir = RETRAIN_MODEL_DIRS[args.corpus]

    if not args.forget_file:
        args.forget_file = DATA_FILES[args.corpus]['forget']
    if not args.retain_file:
        args.retain_file = DATA_FILES[args.corpus]['retain2']
    if not args.holdout_file:
        args.holdout_file = DATA_FILES[args.corpus]['holdout']
    main(**vars(args))

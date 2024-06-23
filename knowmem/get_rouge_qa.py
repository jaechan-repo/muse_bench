import sys
sys.path.append(".")
sys.path.append("../baselines")

from baselines.utils import load_model_and_tokenizer

from transformers import BitsAndBytesConfig
from rouge_score import rouge_scorer
from tqdm.contrib import tzip
import pandas as pd
import torch

import argparse
import os
import random
from typing import List, Dict, Tuple
import json


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def load_in_8bit(*args, **kwargs):
    return load_model_and_tokenizer(
        *args,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=200.0
        ), **kwargs
    )


class EvalLogger:

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=False
        )
        self.history = []


    def log(self, prompt: str, gt: str, output: str, question: str | None = None):
        """Logs the ROUGE scores computed between `gt` and `output`.

        Args:
            prompt (str): Prompt used for creating `output`.
                This argument is only required for the purpose of logging the history.
                No metrics are calculated against it.
            gt (str): Ground truth continuation of the prompt.
            output (str): Model's generated continuation of the `prompt`.
                Warning: The output should NOT begin with the `prompt`.
        """
        score = self.scorer.score(gt, output)
        d = {
            'prompt': prompt,
            'gt': gt,
            'response': output,
            'rougeL': score['rougeL'].fmeasure,
            'rougeL_recall': score['rougeL'].recall,
            'rouge1': score['rouge1'].fmeasure,
            'rouge1_recall': score['rouge1'].recall
        }
        if question is not None: d['question'] = question
        self.history.append(d)


    def report(self) -> Tuple[Dict, Dict]:
        out = {}
        for key, val in self.history[0].items():
            if isinstance(val, str): continue
            vals: List[float] = [item[key] for item in self.history]
            out[f"max_{key}"] = max(vals)
            out[f"mean_{key}"] = sum(vals) / len(vals)
        return out, self.history


def run_completions(model, tokenizer,
                    questions: List[str], answers: List[str],
                    icl_qs: List[str] = [], icl_as: List[str] = [],
                    max_new_tokens : int = 128):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = EvalLogger()
    general_prompt: str = ""

    # Few-shot prompting
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    for question, answer in tzip(questions, answers):
        prompt = general_prompt + f"Question: {question}\nAnswer: "

        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True).input_ids
        assert len(input_ids) == 1

        # Use the `model` to generate the continuation of the `input_ids`.
        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        assert torch.all(output_ids[0].type_as(input_ids[0])[:len(input_ids[0])] == input_ids[0])
        output_ids = output_ids[:, len(input_ids[0]):]

        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()


def main(qa_file: str,
         model_dir: str, tokenizer_dir: str,
         reinforced_model_dir: str | None = None,
         model_name: str | None = None,
         icl_file: str | None = None,
         max_new_tokens: int = 32
         ):
    # Load model and tokenizer.
    model, tokenizer = load_in_8bit(model_dir,
                                    tokenizer_dir=tokenizer_dir,
                                    reinforced_model_dir=reinforced_model_dir,
                                    model_name=model_name)

    # Prepare the dataset.
    with open(qa_file, 'r') as f:
        data = json.load(f)

    if icl_file is not None:
        with open(icl_file, 'r') as f:
            shots = json.load(f)
        icl_qs: List[str] = [shot['question'] for shot in shots]
        icl_as: List[str] = [shot['answer'] for shot in shots]
    else: icl_qs, icl_as = [], []

    questions: List[str] = [d['question'] for d in data]
    answers: List[str] = [d['answer'] for d in data]

    # Run completions and output statistics.
    out, history = run_completions(
        model, tokenizer,
        questions, answers,
        icl_qs, icl_as,
        max_new_tokens=max_new_tokens,
    )
    out |= {
        'model': model_dir,
        'max_new_tokens': max_new_tokens
    }
    return out, history

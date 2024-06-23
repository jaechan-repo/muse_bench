# MUSE-News

MUSE is a comprehensive machine unlearning evaluation benchmark that assesses six key properties for unlearned models: (1) no verbatim memorization, (2) no knowledge memorization, (3) no privacy leakage, (4) utility preservation on data not intended for removal, (5) scalability with respect to the size of removal requests, and (6) sustainability over sequential unlearning requests. MUSE focuses on two types of textual data that commonly require unlearning: news articles (News) and novels (Books). __This repository contains the News corpus of MUSE (MUSE-News), which comprises BBC articles collected post-August 2023__.

## Details on Subsets & Splits

MUSE-News consists of 7 subsets: `raw`, `verbmem`, `knowmem`, `privleak`, `scal`, `sust`, and `train`.
- `raw`: A raw corpus from which all subsets except `scal` and `sust` are derived. The splits are:
    - `forget`: Data intended to be forgotten
    - `retain1`: Data used optionally as a calibrator for unlearning
    - `retain2`: Retain set, i.e. data seen by the target model and used for evaluation
    - `holdout`: Data never seen by the target model during pre-training and unlearning
- `verbmem`: Evaluates __verbatim memorization (C1)__. It contains a single split `forget` with 100 samples verbatim extracted from the `forget` split of the `raw` subset, each up to 2048 tokens long according to LLaMA's tokenization.
- `knowmem`: Evaluates __knowledge memorization (C2)__ and __utility preservation (C4)__. Partitioned into 2 splits: `forget_qa` set (for evaluating forget quality) and `retain_qa` set (for evaluating model utility).  Each split contains 100 question-answer pairs testing the model's knowledge on that specific split of the `raw` subset.
- `scal`: Contains forget sets used to evaluate scalability. The splits are `forget_1`, `forget_2`, `forget_3`, and `forget_4` such that `forget_2` contains `forget_1`, `forget_3` contains `forget_2`, etc.
- `sust`: Contains forget sets used to evaluate sustainability. The splits are `forget_1`, `forget_2`, `forget_3`, and `forget_4` such that all the splits are pair-wise disjoint.
- `train`: Data used for pre-training the target model.

## Loading the datasets

To load the dataset, specify the subset and the split as follows:
```py
from datasets import load_dataset

SUBSET = "verbmem"
SPLIT = "forget"
dataset = load_dataset("muse-bench/MUSE-Books", SUBSET, split=SPLIT)
```

## Applicability

Evaluating with our dataset applies to any unlearning method performed on our [target model](https://huggingface.co/swj0419/bbc-original_STEP0000100_5-31) with respect to the forget set provided in the `raw` subset (or `scal` or `sust` for scalability and sustainability).

## Codebase

For evaluating MIA methods on our datasets, visit our [GitHub repository]().

## Citing our work
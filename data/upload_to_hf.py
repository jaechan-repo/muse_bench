import os
from datasets import DatasetDict, Dataset
import argparse
from huggingface_hub import HfApi, HfFolder


def create_dataset_dict(subset_path: str):
    subset_dict = {}
    for json_file in os.listdir(subset_path):
        if json_file.endswith(".json"):
            split_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(subset_path, json_file)
            subset_dict[split_name] = Dataset.from_json(json_path)
    return DatasetDict(subset_dict)


def upload_to_hf(username: str, token: str, bbc_repo_name: str, hp_repo_name: str):
    HfFolder.save_token(token)
    api = HfApi()
    for data_dir, dataset_name in [('./bbc', bbc_repo_name), ('./hp', hp_repo_name)]:
        try: api.create_repo(dataset_name, token=token)
        except: pass    # repo already exists
        all_subsets = {}
        for subset_dir in os.listdir(data_dir):
            subset_path = os.path.join(data_dir, subset_dir)
            if os.path.isdir(subset_path):
                all_subsets[subset_dir] = create_dataset_dict(subset_path)
        for subset_name, subset_dict in all_subsets.items():
            subset_dict.push_to_hub(f"{username}/{dataset_name}", config_name=subset_name, token=token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_username', type=str, default='muse-bench')
    parser.add_argument('--hf_token', type=str, required=True)
    parser.add_argument('--bbc_repo_name', type=str, default="MUSE-News")
    parser.add_argument('--hp_repo_name', type=str, default="MUSE-Books")
    args = parser.parse_args()
    upload_to_hf(args.hf_username, args.hf_token, args.bbc_repo_name, args.hp_repo_name)
    print("All subsets uploaded successfully!")

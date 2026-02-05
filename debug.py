import numpy as np
import os
from datasets import Dataset, DatasetDict, Features, Sequence, Value

if __name__ == "__main__":
    for dataset_name in ['arc1concept-aug-1000']:
        features = Features({
            "inputs": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            "group": Value("int32"),
        })

        inputs = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/train/all__inputs.npy")
        labels = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/train/all__labels.npy")
        puzzle_indices = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/train/all__puzzle_indices.npy")
        puzzle_identifiers = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/train/all__puzzle_identifiers.npy")
        group_indices = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/train/all__group_indices.npy")

        train_split = Dataset.from_dict({
            "inputs": inputs,
            "labels": labels,
            "group": np.repeat(group_indices[:-1], 8),
        })

        inputs = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/test/all__inputs.npy")
        labels = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/test/all__labels.npy")
        puzzle_indices = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/test/all__puzzle_indices.npy")
        puzzle_identifiers = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/test/all__puzzle_identifiers.npy")
        group_indices = np.load(f"/vol/tmp/goldejon/trm_data/{dataset_name}/test/all__group_indices.npy")

        test_split = Dataset.from_dict({
            "inputs": inputs,
            "labels": labels,
            "group": group_indices[:-1],
        })

        dataset_dict = DatasetDict({
            "train": train_split,
            "test": test_split,
        })
        dataset_dict = dataset_dict.cast(features)

        dataset_dict.push_to_hub('whoisjones/maze')

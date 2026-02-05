from typing import Optional
import math
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    output_dir: str = "/vol/tmp/goldejon/trm_data/maze-30x30-hard-1k"
    subsample_size: Optional[int] = None
    aug: bool = False


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []
    
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)
                
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    # Determine number of augmentations
    if set_name == "train":
        num_augmentations = 8
    else:
        num_augmentations = 1
    
    print(f"\n{'='*60}")
    print(f"Processing {set_name} set:")
    print(f"  Original puzzles: {len(inputs)}")
    print(f"  set_name = '{set_name}'")
    print(f"  Augmentations per puzzle: {num_augmentations}")
    print(f"  Expected total examples: {len(inputs) * num_augmentations}")
    print(f"{'='*60}\n")
    
    for inp, out in zip(tqdm(inputs), labels):
        # Dihedral transformations for augmentation
        for aug_idx in range(num_augmentations):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
        
        # Increment puzzle_id once per original puzzle (not per augmentation)
        puzzle_id += 1
        # Push group
        results["group_indices"].append(puzzle_id)
    
    print(f"Created {len(results['inputs'])} total examples from {len(inputs)} original puzzles")
            
    # Char mappings
    assert len(all_chars - set(CHARSET)) == 0
    
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])
        
        return arr
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    print(f"Final results: inputs shape={results['inputs'].shape}, labels shape={results['labels'].shape}")
    print(f"  puzzle_indices length={len(results['puzzle_indices'])}, group_indices length={len(results['group_indices'])}")

    # Metadata
    # Calculate mean puzzle examples per group
    num_groups = len(results["group_indices"]) - 1
    num_puzzles = len(results["puzzle_indices"]) - 1
    mean_puzzle_examples = num_puzzles / num_groups if num_groups > 0 else 1.0
    
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),  # type: ignore
        vocab_size=len(CHARSET) + 1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num_groups,
        mean_puzzle_examples=mean_puzzle_examples,
        total_puzzles=num_puzzles,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()

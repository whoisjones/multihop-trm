import torch
from torch.utils.data import Sampler
from typing import List, Dict, Iterator
import numpy as np
from collections import defaultdict


class GroupBatchSampler(Sampler):
    """
    Sampler that ensures all examples in a batch come from the same group.
    When a group doesn't have enough examples for a full batch, it combines
    multiple groups to fill the batch.
    
    Ensures all examples are seen during training by tracking which examples
    have been included in batches.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        group_column: str = "group",
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: Dataset with a 'group' column
            batch_size: Desired batch size
            group_column: Name of the column containing group IDs
            shuffle: Whether to shuffle groups and examples within groups
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_column = group_column
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Efficiently extract group IDs - try to use column access for HuggingFace datasets
        try:
            # For HuggingFace datasets, use column access which is much faster
            if hasattr(dataset, 'column_names') and group_column in dataset.column_names:
                group_ids_array = np.array(dataset[group_column])
            else:
                # Fallback: extract groups one by one
                group_ids_array = np.array([dataset[idx][group_column] for idx in range(len(dataset))])
        except Exception:
            # Fallback: extract groups one by one
            group_ids_array = np.array([dataset[idx][group_column] for idx in range(len(dataset))])
        
        # Group examples by group ID using numpy for speed
        self.group_to_indices = defaultdict(list)
        for idx, group_id in enumerate(group_ids_array):
            self.group_to_indices[group_id].append(idx)
        
        # Convert to numpy arrays for faster access
        self.group_to_indices = {
            group_id: np.array(indices, dtype=np.int64)
            for group_id, indices in self.group_to_indices.items()
        }
        
        self.group_ids = list(self.group_to_indices.keys())
        self.num_groups = len(self.group_ids)
        
        # Calculate statistics
        group_sizes = [len(indices) for indices in self.group_to_indices.values()]
        self.total_examples = sum(group_sizes)
        self.avg_group_size = np.mean(group_sizes) if group_sizes else 0
        self.min_group_size = min(group_sizes) if group_sizes else 0
        self.max_group_size = max(group_sizes) if group_sizes else 0
        
        # Pre-compute batches (only once if not shuffling)
        self._batches = self._create_batches()
        
        # Calculate how many batches are pure (single group) vs mixed
        # Track this during batch creation to avoid extra iteration
        self._pure_batches = 0
        self._mixed_batches = 0
    
    def _create_batches(self) -> List[List[int]]:
        """
        Create batches ensuring same-group examples are together.
        When a group doesn't have enough examples, combine with next groups.
        """
        batches = []
        pure_count = 0
        mixed_count = 0
        
        # Shuffle groups if needed
        group_order = self.group_ids.copy()
        if self.shuffle:
            np.random.shuffle(group_order)
        
        # Process groups to create batches
        current_batch = []
        current_batch_groups = set()
        
        for group_id in group_order:
            indices = self.group_to_indices[group_id]
            
            # Shuffle examples within group if needed
            if self.shuffle:
                indices = indices.copy()
                np.random.shuffle(indices)
            
            # If we have a partial batch from previous group(s), try to fill it
            if current_batch:
                # Check if we can complete the batch with current group
                remaining = self.batch_size - len(current_batch)
                
                if len(indices) >= remaining:
                    # Fill the batch with examples from current group
                    batch_indices = list(current_batch) + list(indices[:remaining])
                    batches.append(batch_indices)
                    # Track if batch is pure (single group)
                    # If current_batch_groups contains only current group, it's pure
                    # Otherwise it's mixed (contains previous group + current group)
                    if len(current_batch_groups) == 0 or (len(current_batch_groups) == 1 and group_id in current_batch_groups):
                        pure_count += 1
                    else:
                        mixed_count += 1
                    
                    current_batch = []
                    current_batch_groups = set()
                    indices = indices[remaining:]
                else:
                    # Not enough examples in current group, combine groups
                    current_batch.extend(indices)
                    current_batch_groups.add(group_id)
                    indices = []
                    # Check if combined batch is now full
                    if len(current_batch) >= self.batch_size:
                        # Split: take full batch, keep remainder
                        batches.append(list(current_batch[:self.batch_size]))
                        mixed_count += 1  # Mixed because we combined groups
                        current_batch = list(current_batch[self.batch_size:])
                        if not current_batch:
                            current_batch_groups = set()
                    # Continue to next group to fill batch
                    continue
            
            # Process remaining examples in current group
            while len(indices) >= self.batch_size:
                # Create a full batch from this group
                batches.append(list(indices[:self.batch_size]))
                pure_count += 1
                indices = indices[self.batch_size:]
            
            # Handle remaining examples
            if len(indices) > 0:
                if self.drop_last and len(indices) < self.batch_size:
                    # Drop incomplete batch
                    pass
                else:
                    # Start a new partial batch
                    current_batch = list(indices)
                    current_batch_groups = {group_id}
        
        # Handle final partial batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)
            if len(current_batch_groups) == 1:
                pure_count += 1
            else:
                mixed_count += 1
        
        # Update statistics
        self._pure_batches = pure_count
        self._mixed_batches = mixed_count
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Return an iterator over batches."""
        if self.shuffle:
            # Recreate batches with new random order each epoch
            # Only do this if shuffle is enabled
            self._batches = self._create_batches()
        
        return iter(self._batches)
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._batches)
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics about the sampler."""
        return {
            "num_groups": self.num_groups,
            "total_examples": self.total_examples,
            "avg_group_size": self.avg_group_size,
            "min_group_size": self.min_group_size,
            "max_group_size": self.max_group_size,
            "num_batches": len(self._batches),
            "pure_batches": self._pure_batches,
            "mixed_batches": self._mixed_batches,
            "pure_batch_ratio": self._pure_batches / len(self._batches) if self._batches else 0.0,
        }


def group_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for group-based batching.
    Assumes all examples in the batch are from the same group (enforced by sampler).
    Optimized for performance.
    
    Args:
        batch: List of dataset examples (dicts with 'inputs', 'labels', 'group', etc.)
    
    Returns:
        Batched dictionary with tensors
    """
    if not batch:
        return {}
    
    # Stack tensors for each key - optimized path
    batched = {}
    first_item = batch[0]
    
    for key in first_item.keys():
        # Get all values for this key
        values = [item[key] for item in batch]
        first_value = values[0]
        
        # Fast path for common types
        if isinstance(first_value, torch.Tensor):
            batched[key] = torch.stack(values)
        elif isinstance(first_value, np.ndarray):
            # Use numpy stack then convert - faster than converting each individually
            batched[key] = torch.from_numpy(np.stack(values))
        elif isinstance(first_value, (int, float, np.integer, np.floating)):
            batched[key] = torch.tensor(values, dtype=torch.long if isinstance(first_value, (int, np.integer)) else torch.float)
        elif isinstance(first_value, list):
            # Try to convert list to numpy array first
            try:
                arr = np.array(values)
                batched[key] = torch.from_numpy(arr)
            except (ValueError, TypeError):
                # Fallback: keep as list
                batched[key] = values
        else:
            # Keep as list for other types
            batched[key] = values
    
    return batched


def simple_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Simple collate function for validation/testing.
    Just batches the data without any group-based logic.
    
    Args:
        batch: List of dataset examples (dicts with 'inputs', 'labels', etc.)
    
    Returns:
        Batched dictionary with tensors
    """
    if not batch:
        return {}
    
    batched = {}
    first_item = batch[0]
    
    for key in first_item.keys():
        values = [item[key] for item in batch]
        first_value = values[0]
        
        if isinstance(first_value, torch.Tensor):
            batched[key] = torch.stack(values)
        elif isinstance(first_value, np.ndarray):
            batched[key] = torch.from_numpy(np.stack(values))
        elif isinstance(first_value, (int, float, np.integer, np.floating)):
            batched[key] = torch.tensor(values, dtype=torch.long if isinstance(first_value, (int, np.integer)) else torch.float)
        elif isinstance(first_value, list):
            try:
                arr = np.array(values)
                batched[key] = torch.from_numpy(arr)
            except (ValueError, TypeError):
                batched[key] = values
        else:
            batched[key] = values
    
    return batched


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

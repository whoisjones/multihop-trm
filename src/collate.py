import torch


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        values = [d[k] for d in batch]
        if isinstance(values[0], torch.Tensor) and values[0].ndim > 0:
            out[k] = torch.stack(values)
        else:
            out[k] = torch.tensor(values)
    return out


def infinite_dataloader(dataloader, expected_batch_size=None):
    reference_batch_size = None
    while True:
        for batch in dataloader:
            if not batch:
                continue
                
            first_key = next(iter(batch.keys()))
            actual_batch_size = batch[first_key].shape[0]
            
            if reference_batch_size is None:
                reference_batch_size = actual_batch_size if expected_batch_size is None else expected_batch_size
            
            if actual_batch_size == reference_batch_size:
                yield batch
            # Skip incomplete batches - they'll be included in the next epoch iteration

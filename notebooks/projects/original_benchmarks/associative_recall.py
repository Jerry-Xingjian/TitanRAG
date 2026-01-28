
import torch
from torch.utils.data import Dataset
import random

class AssociativeRecallDataset(Dataset):
    def __init__(self, vocab_size=100, seq_len=128, num_samples=10000, pair_len=2):
        """
        Args:
            vocab_size: Size of the vocabulary (keys and values).
            seq_len: Total length of the sequence.
            num_samples: Number of samples in the dataset.
            pair_len: Number of tokens per key-value pair (usually 2: key, value).
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pair_len = pair_len
        
        # Reserve explicit tokens
        self.QUERY_TOKEN = vocab_size - 1  # Not strictly needed if position implicitly defines query, but good for clarity if we used special tokens
        # For simplicity in this numeric task, we just use the numbers [0, vocab_size-2] for data
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Generate unique keys
        # We need enough keys to fill most of the sequence.
        # Max pairs approx seq_len / 2
        num_pairs = (self.seq_len - 1) // 2 
        
        # Ensure vocab size is sufficient for unique keys if possible, or allow repeats if vocab is small (but AR usually requires unique keys for querying)
        # Let's assume vocab_size >> num_pairs usually, or we sample with replacement but that makes task harder/ambiguous.
        # Standard AR: keys are unique.
        
        available_keys = list(range(self.vocab_size - 10)) # Reserve some top tokens for special use if needed
        if len(available_keys) < num_pairs:
            # If vocab is small, we reduce num_pairs
            num_pairs = len(available_keys)
            
        keys = random.sample(available_keys, num_pairs)
        values = [random.choice(available_keys) for _ in range(num_pairs)]
        
        # 2. Select a target query
        target_idx = random.randint(0, num_pairs - 1)
        query_key = keys[target_idx]
        target_val = values[target_idx]
        
        # 3. Construct sequence
        # input: k1 v1 k2 v2 ... kN vN query_key
        # label: -1 -1 -1 -1 ... -1 -1 target_val
        
        input_ids = []
        labels = []
        
        for k, v in zip(keys, values):
            input_ids.extend([k, v])
            labels.extend([-100, -100]) # Ignore loss for context
            
        # Append query
        input_ids.append(query_key)
        labels.append(target_val)
        
        # Padding if needed (simplified: just truncate or assume fixed size logic matches)
        # For this demo, we return tensors directly.
        # If the generated length < self.seq_len, we might pad (but batching handles it if we use collate).
        # Let's simple return as is, and the model handles variable length or we make it fixed.
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

if __name__ == "__main__":
    ds = AssociativeRecallDataset()
    x, y = ds[0]
    print("Input:", x)
    print("Label:", y)

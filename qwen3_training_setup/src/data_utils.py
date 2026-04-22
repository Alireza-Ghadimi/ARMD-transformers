"""
Data loading and preprocessing utilities for Qwen3 training.
"""

import os
import json
from typing import Optional, Dict, List, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """
    Simple text dataset for causal language modeling.
    
    Handles tokenization and creates input_ids and labels for training.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks (useful for long texts)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts and create chunks
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )
            
            input_ids = encoded["input_ids"]
            
            # Split into chunks if needed
            if len(input_ids) <= max_length:
                self.examples.append(input_ids)
            else:
                # Create overlapping chunks
                for i in range(0, len(input_ids), stride):
                    chunk = input_ids[i:i + max_length]
                    if len(chunk) >= 32:  # Skip very short chunks
                        self.examples.append(chunk)
        
        print(f"Created dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        
        # For causal LM, labels are the same as input_ids (shifted internally by model)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


def load_text_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    text_field: str = "text",
    split: Optional[str] = None,
) -> Dataset:
    """
    Load dataset from various formats.
    
    Supports:
    - .txt files (plain text)
    - .json/.jsonl files (with text field)
    - Hugging Face dataset names
    
    Args:
        data_path: Path to data file or HF dataset name
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_field: Field name containing text (for JSON/HF datasets)
        split: Dataset split to use (for HF datasets)
    
    Returns:
        PyTorch Dataset ready for training
    """
    data_path = Path(data_path)
    
    # Case 1: Plain text file
    if data_path.suffix == ".txt" and data_path.exists():
        print(f"Loading plain text file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split into paragraphs or lines
        texts = [t.strip() for t in text.split("\n\n") if t.strip()]
        if len(texts) == 1:  # If no double newlines, split by single newline
            texts = [t.strip() for t in text.split("\n") if t.strip()]
        
        return TextDataset(texts, tokenizer, max_length)
    
    # Case 2: JSON/JSONL file
    elif data_path.suffix in [".json", ".jsonl"] and data_path.exists():
        print(f"Loading JSON file: {data_path}")
        
        if data_path.suffix == ".jsonl":
            texts = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if text_field in obj:
                        texts.append(obj[text_field])
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                texts = [item[text_field] for item in data if text_field in item]
            elif isinstance(data, dict) and text_field in data:
                texts = [data[text_field]]
            else:
                raise ValueError(f"Cannot find '{text_field}' field in JSON")
        
        return TextDataset(texts, tokenizer, max_length)
    
    # Case 3: Hugging Face dataset
    else:
        print(f"Loading Hugging Face dataset: {data_path}")
        try:
            if split:
                dataset = load_dataset(str(data_path), split=split)
            else:
                dataset = load_dataset(str(data_path))
                # Use first available split
                if isinstance(dataset, dict):
                    split = list(dataset.keys())[0]
                    dataset = dataset[split]
            
            # Extract texts
            if text_field not in dataset.column_names:
                raise ValueError(f"Field '{text_field}' not found. Available: {dataset.column_names}")
            
            texts = dataset[text_field]
            return TextDataset(texts, tokenizer, max_length)
            
        except Exception as e:
            raise ValueError(f"Could not load dataset from '{data_path}': {e}")


def create_sample_dataset(output_path: str, num_samples: int = 100) -> None:
    """
    Create a sample dataset for testing.
    
    Args:
        output_path: Where to save the sample data
        num_samples: Number of samples to generate
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate sample texts
    templates = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was data, and the data was with the model.",
        "Machine learning is the study of computer algorithms that improve through experience.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
    ]
    
    samples = []
    for i in range(num_samples):
        # Create variations
        text = templates[i % len(templates)]
        text = f"Sample {i}: {text} This is training example number {i}."
        samples.append({"text": text})
    
    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created sample dataset: {output_path} ({num_samples} samples)")


class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling.
    
    Handles padding and label masking.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm: bool = False):
        self.tokenizer = tokenizer
        self.mlm = mlm
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_length = max(f["input_ids"].size(0) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            
            # Compute attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            # Mask padding tokens in labels (-100 is ignored by loss)
            labels = labels.masked_fill(~attention_mask.bool(), -100)
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Stack into tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.stack(batch["labels"])
        
        return batch


# Example usage
if __name__ == "__main__":
    print("Data utilities for Qwen3 training")
    print("\nExample usage:")
    print("""
    from transformers import AutoTokenizer
    from data_utils import load_text_dataset, create_sample_dataset
    
    # Create sample data
    create_sample_dataset("data/sample_train.jsonl", num_samples=100)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    # Load dataset
    dataset = load_text_dataset(
        "data/sample_train.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")
    """)

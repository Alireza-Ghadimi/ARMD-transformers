"""
Training utilities and helpers for Qwen3.
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from transformers.trainer_callback import TrainerCallback


@dataclass
class Qwen3TrainingConfig:
    """
    Configuration for Qwen3 training.
    
    This wraps TrainingArguments with sensible defaults for Qwen3.
    """
    
    # Model and data
    model_name: str = "Qwen/Qwen3-0.6B"
    dataset_path: str = "data/train.txt"
    output_dir: str = "./outputs/qwen3_finetuned"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False  # Auto-detected or user-specified
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    
    # Console output control
    disable_tqdm: bool = True
    log_level: str = "error"  # debug, info, warning, error, critical
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "tensorboard"  # or "wandb"
    
    # Advanced
    gradient_checkpointing: bool = False
    max_steps: int = -1  # -1 means use num_train_epochs
    deepspeed: Optional[str] = None  # Path to DeepSpeed config file
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to Transformers TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_grad_norm=self.max_grad_norm,
            optim=self.optim,
            lr_scheduler_type=self.lr_scheduler_type,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            logging_first_step=False,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            disable_tqdm=self.disable_tqdm,
            log_level=self.log_level,
            seed=self.seed,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            report_to=self.report_to,
            gradient_checkpointing=self.gradient_checkpointing,
            max_steps=self.max_steps,
            deepspeed=self.deepspeed,
        )


class PerformanceCallback(TrainerCallback):
    """
    Callback to track training performance metrics.
    """
    
    def __init__(self):
        self.epoch_times = []
        self.step_times = []
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.epoch_start:
            self.epoch_start.record()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start and torch.cuda.is_available():
            epoch_end = torch.cuda.Event(enable_timing=True)
            epoch_end.record()
            torch.cuda.synchronize()
            elapsed = self.epoch_start.elapsed_time(epoch_end) / 1000  # Convert to seconds
            self.epoch_times.append(elapsed)
            print(f"\n⏱ Epoch completed in {elapsed:.2f}s")


def setup_model_for_training(
    model: PreTrainedModel,
    gradient_checkpointing: bool = False,
    use_cache: bool = False,
) -> PreTrainedModel:
    """
    Prepare model for training.
    
    Args:
        model: The model to prepare
        gradient_checkpointing: Enable gradient checkpointing to save memory
        use_cache: Whether to use KV cache (should be False for training)
    
    Returns:
        Prepared model
    """
    # Disable cache during training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # Ensure model is in training mode
    model.train()
    
    return model


def print_training_info(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Qwen3TrainingConfig,
    dataset_size: int,
) -> None:
    """Print training configuration and model information."""
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Training steps
    steps_per_epoch = dataset_size // (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.num_train_epochs if config.max_steps == -1 else config.max_steps
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {config.model_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"\nDataset: {config.dataset_path}")
    print(f"  Size: {dataset_size:,} examples")
    print(f"\nTraining:")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")
    print(f"\nOptimization:")
    print(f"  Optimizer: {config.optim}")
    print(f"  Scheduler: {config.lr_scheduler_type}")
    print(f"  Mixed precision: {'fp16' if config.fp16 else 'bf16' if config.bf16 else 'fp32'}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"\nOutput: {config.output_dir}")
    print(f"Logging: {config.report_to}")
    print("="*70 + "\n")


def save_training_summary(
    output_dir: str,
    model_name: str,
    config: Qwen3TrainingConfig,
    final_loss: Optional[float] = None,
    modified_layers: Optional[list] = None,
) -> None:
    """
    Save a training summary to a text file.
    
    Useful for record-keeping and reproducibility.
    """
    output_path = Path(output_dir) / "training_summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("Qwen3 Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {config.dataset_path}\n")
        f.write(f"\nTraining Configuration:\n")
        f.write(f"  Epochs: {config.num_train_epochs}\n")
        f.write(f"  Batch size: {config.per_device_train_batch_size}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Optimizer: {config.optim}\n")
        f.write(f"  Scheduler: {config.lr_scheduler_type}\n")
        
        if modified_layers:
            f.write(f"\nModified Layers: {modified_layers}\n")
        
        if final_loss is not None:
            f.write(f"\nFinal Loss: {final_loss:.4f}\n")
        
        f.write(f"\nOutput Directory: {output_dir}\n")
    
    print(f"✓ Training summary saved to {output_path}")


def estimate_memory_usage(
    model: PreTrainedModel,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing: bool = False,
) -> Dict[str, float]:
    """
    Estimate GPU memory usage for training.
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    
    # Gradients (same size as parameters)
    gradient_memory = param_memory
    
    # Optimizer states (Adam: 2x parameters for momentum and variance)
    optimizer_memory = 2 * param_memory
    
    # Activations (rough estimate)
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    activation_memory = (
        batch_size * sequence_length * hidden_size * num_layers * 4  # 4 bytes per float32
    ) / 1e9
    
    # Gradient checkpointing reduces activation memory
    if gradient_checkpointing:
        activation_memory *= 0.3  # Rough estimate
    
    # Total
    total = param_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        "parameters": param_memory,
        "gradients": gradient_memory,
        "optimizer": optimizer_memory,
        "activations": activation_memory,
        "total": total,
    }


# Example usage
if __name__ == "__main__":
    print("Training utilities for Qwen3")
    print("\nExample usage:")
    print("""
    from training_utils import Qwen3TrainingConfig, setup_model_for_training
    from transformers import Qwen3ForCausalLM, Trainer
    
    # Create config
    config = Qwen3TrainingConfig(
        model_name="Qwen/Qwen3-0.6B",
        dataset_path="data/train.txt",
        output_dir="./outputs/my_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
    )
    
    # Load and prepare model
    model = Qwen3ForCausalLM.from_pretrained(config.model_name)
    model = setup_model_for_training(model, gradient_checkpointing=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=config.to_training_arguments(),
        train_dataset=train_dataset,
        # ... other arguments
    )
    
    # Train
    trainer.train()
    """)

#!/usr/bin/env python3
"""
Full fine-tuning script for Qwen3-0.6B.

This script performs standard full fine-tuning on the entire model
using the official Transformers Qwen3 implementation.
"""

import sys
import os
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as transformers_logging

from data_utils import load_text_dataset, DataCollatorForCausalLM, create_sample_dataset
from training_utils import (
    Qwen3TrainingConfig,
    setup_model_for_training,
    print_training_info,
    save_training_summary,
    PerformanceCallback,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Full fine-tuning for Qwen3-0.6B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model and data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Pretrained model name or path")
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="Path to training data or HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use (for HuggingFace datasets)")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Field name for text in JSON/HF datasets")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large HuggingFace datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (useful for testing)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_finetuned",
                        help="Output directory for checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (-1 = use epochs)")
    
    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 mixed precision (auto-detected if not specified)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 mixed precision")
    parser.add_argument("--fp32", action="store_true",
                        help="Use float32 (no mixed precision)")
    
    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--report_to", type=str, default="wandb",
                        choices=["tensorboard", "wandb", "none"],
                        help="Logging platform")
    parser.add_argument("--disable_tqdm", action="store_true", default=True,
                        help="Disable progress bars in console")
    parser.add_argument("--enable_tqdm", action="store_true",
                        help="Enable progress bars in console (overrides --disable_tqdm)")
    parser.add_argument("--log_level", type=str, default="error",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Console logging level (warning/error reduces output)")
    
    # Wandb configuration
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity (team/username) to log to")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create sample dataset if data doesn't exist")
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config file (e.g., ds_config_zero3.json)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (automatically set by DeepSpeed)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Handle tqdm override
    if args.enable_tqdm:
        args.disable_tqdm = False
    
    # Configure logging based on log_level argument
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    
    # Set logging level for all loggers
    logging.basicConfig(
        level=log_level_map[args.log_level],
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set transformers logging level
    transformers_logging.set_verbosity(log_level_map[args.log_level])
    
    # Disable default transformers logging to console if log_level is warning or higher
    if args.log_level in ["warning", "error", "critical"]:
        transformers_logging.disable_default_handler()
        transformers_logging.disable_progress_bar()
    
    # Auto-detect and configure mixed precision
    if not any([args.bf16, args.fp16, args.fp32]):
        # No precision specified, auto-detect
        if torch.cuda.is_available():
            # Check if bf16 is supported
            try:
                # Try to create a bf16 tensor on GPU to test support
                test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                args.bf16 = True
                print("Auto-detected: bf16 supported, using bf16")
            except:
                # Fall back to fp16
                args.fp16 = True
                print("Auto-detected: bf16 not supported, using fp16")
        else:
            # CPU training
            args.fp32 = True
            print("No GPU detected, using fp32")
    else:
        # Ensure only one precision mode is enabled
        precision_modes = sum([args.bf16, args.fp16, args.fp32])
        if precision_modes > 1:
            print("⚠ Warning: Multiple precision modes specified. Using fp16.")
            args.bf16 = False
            args.fp16 = True
            args.fp32 = False
    
    print("\n" + "="*70)
    print("QWEN3-0.6B FULL FINE-TUNING")
    print("="*70 + "\n")
    
    # Initialize wandb if selected
    if args.report_to == "wandb":
        try:
            import wandb
            
             
            # Set wandb environment variables if provided
            if args.wandb_entity:
                os.environ["WANDB_ENTITY"] = args.wandb_entity
                print(f"Wandb entity set to: {args.wandb_entity}")
            
            if args.wandb_project:
                os.environ["WANDB_PROJECT"] = args.wandb_project
                print(f"Wandb project set to: {args.wandb_project}")
            
            if args.wandb_run_name:
                os.environ["WANDB_NAME"] = args.wandb_run_name
                print(f"Wandb run name set to: {args.wandb_run_name}")
            
            print("Weights & Biases logging enabled")
            print("  Make sure you're logged in: wandb login")
            
            # Check if entity is set
            entity = os.environ.get("WANDB_ENTITY")
            if entity:
                print(f"  Logging to entity: {entity}")
            else:
                print("  ⚠ No WANDB_ENTITY set - may fail if personal entities are disabled")
                print("  Set with: --wandb_entity YOUR_TEAM_NAME or export WANDB_ENTITY=YOUR_TEAM_NAME")
            
            print(f"  View at: https://wandb.ai\n")
        except ImportError:
            print("⚠ Warning: wandb not installed. Install with: pip install wandb")
            print("  Falling back to tensorboard logging\n")
            args.report_to = "tensorboard"
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token")
    
    # Load dataset - handle both HuggingFace datasets and local files
    print(f"\nLoading dataset: {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
    
    # Check if it's a local file or HuggingFace dataset
    if dataset_path.exists():
        # Local file
        train_dataset = load_text_dataset(
            args.dataset_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            text_field=args.text_field,
        )
    else:
        # Assume it's a HuggingFace dataset
        print(f"  Loading from HuggingFace Hub: {args.dataset_path}")
        if args.streaming:
            print(f"  Using streaming mode")
        
        from datasets import load_dataset
        
        # Load dataset with optional streaming and split
        dataset = load_dataset(
            args.dataset_path,
            split=args.dataset_split,
            streaming=args.streaming,
        )
        
        # Limit samples if specified
        if args.max_samples is not None:
            print(f"  Limiting to {args.max_samples} samples")
            if args.streaming:
                dataset = dataset.take(args.max_samples)
            else:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
        # Extract texts and create TextDataset
        if args.text_field not in dataset.column_names:
            raise ValueError(f"Field '{args.text_field}' not found. Available: {dataset.column_names}")
        
        # Convert to list for non-streaming, or take samples for streaming
        if args.streaming:
            texts = [item[args.text_field] for item in dataset]
        else:
            texts = dataset[args.text_field]
        
        from data_utils import TextDataset
        train_dataset = TextDataset(texts, tokenizer, args.max_length)
    
    print(f"  Dataset size: {len(train_dataset)} examples")
    
    # Load model with pretrained weights
    print(f"\nLoading model: {args.model_name}")
    print(f"  Loading pretrained weights from HuggingFace Hub...")
    
    # Determine torch dtype based on precision settings
    if args.bf16:
        torch_dtype = torch.bfloat16
        print(f"  Using bfloat16 precision")
    elif args.fp16:
        torch_dtype = torch.float16
        print(f"  Using float16 precision")
    else:
        torch_dtype = torch.float32
        print(f"  Using float32 precision")
    
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,  # Let Trainer handle device placement
        trust_remote_code=True,
    )
    
    print(f"  ✓ Pretrained weights loaded successfully")
    
    # Verify all parameters are trainable for full fine-tuning
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    if trainable_params != total_params:
        print(f"  ⚠ Warning: Not all parameters are trainable!")
        print(f"  Enabling all parameters for full fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ All {trainable_params:,} parameters now trainable")
    else:
        print(f"  ✓ Full fine-tuning mode: All parameters trainable")
    
    # Prepare model for training
    model = setup_model_for_training(
        model,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cache=False,
    )
    
    # Create training config
    config = Qwen3TrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        disable_tqdm=args.disable_tqdm,
        log_level=args.log_level,
    )
    
    # Print training info
    print_training_info(model, tokenizer, config, len(train_dataset))
    
    # Create data collator
    data_collator = DataCollatorForCausalLM(tokenizer)
    
    # Create trainer
    training_args = config.to_training_arguments()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[PerformanceCallback()],
    )
    
    # Train
    print("Starting training...")
    print(f"  Mode: Full fine-tuning (all {trainable_params:,} parameters)")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Output directory: {args.output_dir}")
    if args.report_to == "wandb":
        print(f"  Logging: Weights & Biases (wandb)")
        print(f"  View at: https://wandb.ai")
    elif args.report_to == "tensorboard":
        print(f"  Logging: TensorBoard")
        print(f"  View with: tensorboard --logdir {args.output_dir}/runs")
    print()
    
    try:
        train_result = trainer.train()
        
        # Save final model
        print("\nTraining completed! Saving model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training summary
        final_loss = train_result.metrics.get("train_loss", None)
        save_training_summary(
            args.output_dir,
            args.model_name,
            config,
            final_loss=final_loss,
        )
        
        print(f"\n✓ Model saved to: {args.output_dir}")
        print(f"✓ Final training loss: {final_loss:.4f}" if final_loss else "")
        
        # Print next steps
        print("\n" + "="*70)
        print("Training complete! Next steps:")
        print("="*70)
        if args.report_to == "wandb":
            print(f"1. View training logs at: https://wandb.ai")
        else:
            print(f"1. View logs: tensorboard --logdir {args.output_dir}/runs")
        print(f"2. Test model: python scripts/inference_test.py")
        print(f"3. Use model:")
        print(f"   from transformers import AutoTokenizer, Qwen3ForCausalLM")
        print(f"   model = Qwen3ForCausalLM.from_pretrained('{args.output_dir}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial checkpoints saved in: {args.output_dir}")
        return 1
    
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

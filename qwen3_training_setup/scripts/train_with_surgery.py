#!/usr/bin/env python3
"""
Training script with optional layer surgery for Qwen3-0.6B.

This script allows you to:
1. Replace specific decoder layers with custom implementations
2. Load pretrained weights for unchanged layers
3. Train only modified layers (optional)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
)

from data_utils import load_text_dataset, DataCollatorForCausalLM, create_sample_dataset
from training_utils import (
    Qwen3TrainingConfig,
    setup_model_for_training,
    print_training_info,
    save_training_summary,
    PerformanceCallback,
)
from model_wrapper import (
    replace_decoder_layers,
    CustomQwen3DecoderLayer,
    get_layer_info,
    freeze_layers,
    count_parameters,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3 training with optional layer surgery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model and data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Pretrained model name or path")
    parser.add_argument("--dataset_path", type=str, default="data/train.txt",
                        help="Path to training data")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Layer surgery
    parser.add_argument("--replace_layers", type=str, default="",
                        help="Comma-separated layer indices to replace (e.g., '12,13,14')")
    parser.add_argument("--freeze_unchanged_layers", action="store_true",
                        help="Freeze all layers except modified ones")
    parser.add_argument("--custom_tag", type=str, default="modified",
                        help="Tag for custom layers (for identification)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_surgery",
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
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (-1 = use epochs)")
    
    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 mixed precision")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="Logging platform")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create sample dataset if data doesn't exist")
    
    return parser.parse_args()


def main():
    """Main training function with layer surgery."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("QWEN3-0.6B TRAINING WITH LAYER SURGERY")
    print("="*70 + "\n")
    
    # Parse layer indices
    if args.replace_layers:
        replace_layer_indices = [int(x.strip()) for x in args.replace_layers.split(",")]
        print(f"Will replace layers: {replace_layer_indices}")
    else:
        replace_layer_indices = []
        print("No layers will be replaced (standard training)")
    
    # Check if dataset exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        if args.create_sample_data:
            print(f"\nDataset not found. Creating sample data at {dataset_path}")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            create_sample_dataset(dataset_path, num_samples=100)
        else:
            print(f"\nError: Dataset not found at {dataset_path}")
            print("Use --create_sample_data to create a sample dataset")
            return 1
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset_path}")
    train_dataset = load_text_dataset(
        args.dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"  Dataset size: {len(train_dataset)} examples")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map=None,
    )
    
    print("\nOriginal model structure:")
    get_layer_info(model)
    
    # Perform layer surgery if requested
    if replace_layer_indices:
        print(f"\nPerforming layer surgery...")
        print(f"  Replacing {len(replace_layer_indices)} layers")
        
        model = replace_decoder_layers(
            model,
            layer_indices=replace_layer_indices,
            replacement_layer_class=CustomQwen3DecoderLayer,
            custom_kwargs={"custom_tag": args.custom_tag},
        )
        
        print("\nModified model structure:")
        get_layer_info(model)
        
        # Optionally freeze unchanged layers
        if args.freeze_unchanged_layers:
            all_layers = set(range(model.config.num_hidden_layers))
            frozen_layers = list(all_layers - set(replace_layer_indices))
            
            print(f"\nFreezing unchanged layers: {frozen_layers}")
            model = freeze_layers(model, frozen_layers)
    
    # Print parameter counts
    param_counts = count_parameters(model, trainable_only=False)
    trainable_counts = count_parameters(model, trainable_only=True)
    
    print(f"\nParameter summary:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {trainable_counts['total']:,}")
    print(f"  Trainable %: {100 * trainable_counts['total'] / param_counts['total']:.1f}%")
    
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
        max_steps=args.max_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
    )
    
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
    print("Starting training with layer modifications...")
    print(f"  Modified layers: {replace_layer_indices if replace_layer_indices else 'None'}")
    print(f"  Output directory: {args.output_dir}\n")
    
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
            modified_layers=replace_layer_indices,
        )
        
        print(f"\n✓ Model saved to: {args.output_dir}")
        print(f"✓ Modified layers: {replace_layer_indices}")
        print(f"✓ Final training loss: {final_loss:.4f}" if final_loss else "")
        
        # Print next steps
        print("\n" + "="*70)
        print("Training complete!")
        print("="*70)
        print(f"Modified model saved with {len(replace_layer_indices)} custom layer(s)")
        print("\nTo use this model:")
        print(f"  from transformers import Qwen3ForCausalLM, AutoTokenizer")
        print(f"  model = Qwen3ForCausalLM.from_pretrained('{args.output_dir}')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
        print("\nNote: This model contains custom layers and may not be")
        print("compatible with standard Qwen3 implementations.")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

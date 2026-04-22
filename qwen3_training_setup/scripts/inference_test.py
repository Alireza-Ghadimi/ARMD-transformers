#!/usr/bin/env python3
"""
Inference smoke test for Qwen3-0.6B.

This script tests:
1. Tokenizer loading
2. Model loading with pretrained weights
3. Forward pass
4. Text generation

Run this first to verify your setup works before training.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, Qwen3ForCausalLM
from model_wrapper import get_layer_info, count_parameters


def test_tokenizer(model_name: str = "Qwen/Qwen3-0.6B"):
    """Test tokenizer loading and basic functionality."""
    print("\n" + "="*70)
    print("TEST 1: Tokenizer")
    print("="*70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Model max length: {tokenizer.model_max_length}")
        print(f"  Padding side: {tokenizer.padding_side}")
        
        # Test encoding
        test_text = "Hello, how are you?"
        encoded = tokenizer(test_text, return_tensors="pt")
        print(f"\n  Test encoding: '{test_text}'")
        print(f"  Input IDs shape: {encoded['input_ids'].shape}")
        print(f"  Input IDs: {encoded['input_ids'].tolist()}")
        
        # Test decoding
        decoded = tokenizer.decode(encoded['input_ids'][0])
        print(f"  Decoded: '{decoded}'")
        
        return tokenizer
        
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return None


def test_model_loading(model_name: str = "Qwen/Qwen3-0.6B"):
    """Test model loading with pretrained weights."""
    print("\n" + "="*70)
    print("TEST 2: Model Loading")
    print("="*70)
    
    try:
        print(f"Loading model: {model_name}")
        print("(This may take a minute on first run - downloading weights...)")
        
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        
        print(f"✓ Model loaded: {model.__class__.__name__}")
        
        # Print model info
        get_layer_info(model)
        
        # Count parameters
        param_counts = count_parameters(model)
        print(f"Parameter counts:")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Embeddings: {param_counts['embeddings']:,}")
        print(f"  Transformer layers: {param_counts['layers']:,}")
        print(f"  LM head: {param_counts['lm_head']:,}")
        
        # Check device
        device = next(model.parameters()).device
        print(f"\n  Device: {device}")
        print(f"  Dtype: {next(model.parameters()).dtype}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, tokenizer):
    """Test model forward pass."""
    print("\n" + "="*70)
    print("TEST 3: Forward Pass")
    print("="*70)
    
    try:
        # Prepare input
        test_text = "The capital of France is"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"Input text: '{test_text}'")
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  Logits dtype: {outputs.logits.dtype}")
        
        # Get predicted next token
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode([next_token_id])
        
        print(f"  Predicted next token: '{next_token}' (ID: {next_token_id})")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model, tokenizer):
    """Test text generation."""
    print("\n" + "="*70)
    print("TEST 4: Text Generation")
    print("="*70)
    
    try:
        # Test prompts
        prompts = [
            "The capital of France is",
            "Once upon a time",
            "To be or not to be,",
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nGeneration {i}:")
            print(f"  Prompt: '{prompt}'")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Generated: '{generated_text}'")
        
        print(f"\n✓ Text generation successful")
        return True
        
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "#"*70)
    print("# QWEN3-0.6B INFERENCE SMOKE TEST")
    print("#"*70)
    
    model_name = "Qwen/Qwen3-0.6B"
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run tests
    results = []
    
    # Test 1: Tokenizer
    tokenizer = test_tokenizer(model_name)
    results.append(("Tokenizer", tokenizer is not None))
    
    if tokenizer is None:
        print("\n✗ Cannot proceed without tokenizer")
        return
    
    # Test 2: Model
    model = test_model_loading(model_name)
    results.append(("Model Loading", model is not None))
    
    if model is None:
        print("\n✗ Cannot proceed without model")
        return
    
    # Test 3: Forward pass
    forward_ok = test_forward_pass(model, tokenizer)
    results.append(("Forward Pass", forward_ok))
    
    # Test 4: Generation
    generation_ok = test_generation(model, tokenizer)
    results.append(("Text Generation", generation_ok))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("  1. Prepare your training data in data/")
        print("  2. Run: python scripts/train.py --help")
        print("  3. Start training!")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

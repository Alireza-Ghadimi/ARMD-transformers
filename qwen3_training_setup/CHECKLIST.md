# Qwen3-0.6B Training Setup - Verification Checklist

## ✅ Pre-Setup Checklist

Before you begin, verify:

- [ ] Python 3.10+ is installed
  ```bash
  python --version
  ```

- [ ] UV is installed
  ```bash
  uv --version
  # If not: pip install uv
  ```

- [ ] You have access to GPU (optional but recommended)
  ```bash
  nvidia-smi
  ```

- [ ] You have internet connection (for downloading model)

- [ ] You have sufficient disk space (~5GB for model + env)
  ```bash
  df -h .
  ```

## ✅ Setup Verification Checklist

After running setup commands, verify:

### 1. Environment Created
- [ ] Virtual environment exists
  ```bash
  ls -la .venv/
  ```

### 2. Dependencies Installed
- [ ] PyTorch installed
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

- [ ] Transformers installed
  ```bash
  python -c "import transformers; print(transformers.__version__)"
  ```

- [ ] CUDA available (if using GPU)
  ```bash
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
  ```

### 3. Project Structure
- [ ] All source files exist
  ```bash
  ls src/model_wrapper.py
  ls src/data_utils.py
  ls src/training_utils.py
  ```

- [ ] All scripts exist
  ```bash
  ls scripts/inference_test.py
  ls scripts/train.py
  ls scripts/train_with_surgery.py
  ```

- [ ] Config files exist
  ```bash
  ls configs/training_config.yaml
  ls configs/surgery_config.yaml
  ```

### 4. Smoke Test Passes
- [ ] Run inference test
  ```bash
  python scripts/inference_test.py
  ```

Expected output: All 4 tests PASS
- ✓ Tokenizer
- ✓ Model Loading
- ✓ Forward Pass
- ✓ Text Generation

## ✅ Training Preparation Checklist

Before starting training:

### 1. Data Ready
- [ ] Training data exists in `data/` directory
  ```bash
  ls data/
  ```

- [ ] Data format is correct (txt, json, or jsonl)

- [ ] OR using `--create_sample_data` flag for testing

### 2. GPU Resources
- [ ] GPU allocated (if using SLURM)
  ```bash
  squeue -u $USER  # Check allocation
  nvidia-smi       # Verify GPU access
  ```

- [ ] Virtual environment activated
  ```bash
  which python  # Should show .venv path
  ```

### 3. Output Directory
- [ ] Output directory specified and writable
  ```bash
  mkdir -p outputs/test_run
  ```

### 4. Training Configuration
- [ ] Batch size appropriate for GPU memory
  - 8GB GPU: use `--per_device_train_batch_size 1 --gradient_checkpointing`
  - 16GB GPU: use `--per_device_train_batch_size 4`
  - 24GB+ GPU: use `--per_device_train_batch_size 8`

- [ ] Learning rate set appropriately
  - Default: `2e-5` (good starting point)
  - Larger models/datasets: `1e-5` to `5e-5`

## ✅ Training Execution Checklist

During training:

### 1. Training Started Successfully
- [ ] No immediate errors
- [ ] Loss is logged
- [ ] Progress bar shows

### 2. Monitoring Active
- [ ] TensorBoard running (optional)
  ```bash
  tensorboard --logdir outputs/qwen3_finetuned/runs
  ```

- [ ] Can view logs in browser

### 3. Checkpoints Saving
- [ ] Checkpoints appear in output directory
  ```bash
  ls outputs/qwen3_finetuned/
  ```

## ✅ Post-Training Checklist

After training completes:

### 1. Training Completed
- [ ] Training finished without errors
- [ ] Final loss logged
- [ ] Model saved to output directory

### 2. Output Files Present
- [ ] Model weights: `pytorch_model.bin` or `model.safetensors`
  ```bash
  ls outputs/qwen3_finetuned/pytorch_model.bin
  # OR
  ls outputs/qwen3_finetuned/model.safetensors
  ```

- [ ] Config file: `config.json`
  ```bash
  ls outputs/qwen3_finetuned/config.json
  ```

- [ ] Tokenizer files
  ```bash
  ls outputs/qwen3_finetuned/tokenizer*
  ```

- [ ] Training summary
  ```bash
  cat outputs/qwen3_finetuned/training_summary.txt
  ```

### 3. Model Loadable
- [ ] Can load trained model
  ```python
  from transformers import Qwen3ForCausalLM
  model = Qwen3ForCausalLM.from_pretrained("outputs/qwen3_finetuned")
  print("✓ Model loaded successfully")
  ```

### 4. Model Generates Text
- [ ] Can generate with trained model
  ```python
  from transformers import Qwen3ForCausalLM, AutoTokenizer
  
  model = Qwen3ForCausalLM.from_pretrained("outputs/qwen3_finetuned")
  tokenizer = AutoTokenizer.from_pretrained("outputs/qwen3_finetuned")
  
  inputs = tokenizer("Test:", return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=20)
  print(tokenizer.decode(outputs[0]))
  ```

## ✅ Layer Surgery Verification (If Used)

If you used `train_with_surgery.py`:

### 1. Layer Modifications Applied
- [ ] Modified layers logged during training
- [ ] Training summary shows modified layers
  ```bash
  cat outputs/qwen3_modified/training_summary.txt
  ```

### 2. Model Structure Correct
- [ ] Can inspect layer structure
  ```python
  from transformers import Qwen3ForCausalLM
  from src.model_wrapper import get_layer_info
  
  model = Qwen3ForCausalLM.from_pretrained("outputs/qwen3_modified")
  get_layer_info(model)
  # Should show [MODIFIED] markers for replaced layers
  ```

### 3. Custom Layers Functional
- [ ] Model still generates text
- [ ] No errors during forward pass
- [ ] Gradient flow works (if continuing training)

## 🎯 Success Criteria

Your setup is fully working if:

✅ All smoke tests pass
✅ Training runs without errors
✅ Model saves successfully
✅ Trained model can be loaded and used
✅ Text generation works with trained model

## 🐛 Common Issues Quick Fix

| Issue | Quick Fix |
|-------|-----------|
| CUDA out of memory | Add `--gradient_checkpointing --per_device_train_batch_size 1` |
| Module not found | Run `uv pip install -e .` again |
| Tokenizer error | Ensure using AutoTokenizer, not specific class |
| Model download fails | Check internet, try `export HF_HOME=/tmp/huggingface` |
| Import errors | Verify `source .venv/bin/activate` ran |

## 📊 Expected Training Times (Approximate)

On a single GPU (A100/H100):
- Sample data (100 examples, 3 epochs): ~2-5 minutes
- Small dataset (1K examples, 3 epochs): ~10-20 minutes
- Medium dataset (10K examples, 3 epochs): ~1-2 hours
- Large dataset (100K examples, 3 epochs): ~10-20 hours

Times vary based on:
- GPU type and memory
- Sequence length
- Batch size
- Gradient accumulation steps

## 🎓 Learning Path

Recommended progression:

1. ✅ Run smoke test → Verify basic setup
2. ✅ Train on sample data → Test training loop
3. ✅ Train on real data (small) → Verify data pipeline
4. ✅ Experiment with hyperparameters → Optimize performance
5. ✅ Try layer surgery → Explore architecture modifications
6. ✅ Scale to larger datasets → Production training

## 📝 Notes

- Keep this checklist handy as you work
- Check off items as you complete them
- If any item fails, refer to SETUP.md for detailed instructions
- For layer surgery details, see configs/surgery_config.yaml

---

**Ready to start? Begin with the Setup Verification Checklist above! 🚀**

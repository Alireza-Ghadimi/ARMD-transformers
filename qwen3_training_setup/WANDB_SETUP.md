# Weights & Biases (wandb) Setup Guide

## Quick Start

### 1. Install wandb (included in dependencies)
```bash
# Already included in pyproject.toml
uv sync  # This installs wandb automatically
```

### 2. Login to wandb
```bash
wandb login
```

You'll be prompted to enter your API key. Get it from: https://wandb.ai/authorize

Alternatively, set your API key as an environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. Run training (wandb is now default)
```bash
python scripts/train.py --max_samples 10000
```

## Viewing Your Training Runs

Once training starts:
1. Visit https://wandb.ai
2. Go to your project (default: "qwen3-training-setup")
3. View real-time metrics, loss curves, GPU usage, and more

## Customizing wandb Settings

### Change Project Name
```bash
export WANDB_PROJECT=my_qwen3_project
python scripts/train.py --max_samples 10000
```

### Change Run Name
```bash
export WANDB_NAME=fineweb_experiment_1
python scripts/train.py --max_samples 10000
```

### Disable wandb (use tensorboard instead)
```bash
python scripts/train.py --report_to tensorboard
```

### Run in Offline Mode
```bash
export WANDB_MODE=offline
python scripts/train.py --max_samples 10000
```

Later sync offline runs:
```bash
wandb sync wandb/offline-run-*
```

## Environment Variables

Common wandb environment variables:

```bash
# Project name
export WANDB_PROJECT=qwen3-fineweb-training

# Run name/ID
export WANDB_NAME=experiment_01

# Entity (team/username)
export WANDB_ENTITY=your_username

# Disable wandb entirely
export WANDB_DISABLED=true

# Run offline
export WANDB_MODE=offline

# Silent mode (less console output)
export WANDB_SILENT=true
```

## What Gets Logged

By default, wandb logs:
- ✅ Training loss
- ✅ Learning rate
- ✅ Gradient norms
- ✅ Training steps/epoch
- ✅ GPU memory usage
- ✅ Training time
- ✅ Model hyperparameters
- ✅ System info (GPU, CPU, etc.)

## Advanced: Custom Logging

To add custom metrics, modify the training script:

```python
# In your training script
import wandb

# Log custom metrics
wandb.log({
    "custom_metric": value,
    "another_metric": other_value,
})
```

## Troubleshooting

### Issue: "wandb: ERROR Error uploading"
**Solution**: Check internet connection or run in offline mode

### Issue: "wandb: ERROR API key not configured"
**Solution**: Run `wandb login` or set `WANDB_API_KEY`

### Issue: Too much console output
**Solution**: Set `export WANDB_SILENT=true`

### Issue: Want to use tensorboard instead
**Solution**: Use `--report_to tensorboard` flag

## Team Collaboration

### Share runs with your team:
1. Create a team at https://wandb.ai/teams
2. Set your entity: `export WANDB_ENTITY=team_name`
3. Run training - results automatically sync to team workspace

### Share a specific run:
1. Go to your run on wandb.ai
2. Click "Share" button
3. Get shareable link or add collaborators

## Best Practices

1. **Name your runs meaningfully**:
   ```bash
   export WANDB_NAME=qwen3_fineweb_lr2e5_bs4
   ```

2. **Tag experiments**:
   ```bash
   export WANDB_TAGS=baseline,fineweb,qwen3
   ```

3. **Add notes**:
   ```bash
   export WANDB_NOTES="Testing learning rate 2e-5 on fineweb-edu"
   ```

4. **Group related runs**:
   ```bash
   export WANDB_RUN_GROUP=fineweb_experiments
   ```

## Cost and Limits

- **Free tier**: Unlimited experiments, 100GB storage
- **Academic**: Free unlimited storage for students/researchers
- **Teams**: Paid plans for larger teams

Apply for academic account: https://wandb.ai/academic

## More Information

- Documentation: https://docs.wandb.ai
- Examples: https://github.com/wandb/examples
- Community: https://community.wandb.ai

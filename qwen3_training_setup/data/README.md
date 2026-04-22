# Training Data Directory

Place your training data files here.

## Supported Formats

### 1. Plain Text (`.txt`)
Simple text file with your training data.

**Example**: `train.txt`
```
This is the first paragraph of training text.
It can be multiple sentences.

This is the second paragraph.
More training data here.

Each double newline creates a new training example.
```

### 2. JSON (`.json`)
JSON file with a "text" field (or custom field name).

**Example**: `train.json`
```json
[
  {"text": "First training example."},
  {"text": "Second training example."},
  {"text": "Third training example."}
]
```

### 3. JSONL (`.jsonl`)
One JSON object per line.

**Example**: `train.jsonl`
```json
{"text": "First training example."}
{"text": "Second training example."}
{"text": "Third training example."}
```

## Creating Sample Data

For quick testing, create sample data:

```bash
python -c "
from src.data_utils import create_sample_dataset
create_sample_dataset('data/sample_train.jsonl', num_samples=100)
"
```

Or use the `--create_sample_data` flag with training scripts.

## Data Preparation Tips

### 1. Cleaning
- Remove or handle special characters
- Ensure consistent encoding (UTF-8)
- Remove excessive whitespace

### 2. Formatting
- Keep examples reasonably sized (< 2048 tokens)
- Use clear separators (double newlines for .txt)
- Include diverse examples

### 3. Size Guidelines
- **Quick test**: 100-1,000 examples
- **Development**: 1,000-10,000 examples
- **Production**: 10,000+ examples

### 4. Quality over Quantity
- High-quality data yields better results
- Diverse examples improve generalization
- Remove duplicates and low-quality text

## Example: Converting Custom Data

### From CSV to JSONL

```python
import csv
import json

with open('data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    with open('data/train.jsonl', 'w') as jsonlfile:
        for row in reader:
            # Assuming CSV has a 'content' column
            obj = {"text": row['content']}
            jsonlfile.write(json.dumps(obj) + '\n')
```

### From Multiple Text Files

```python
import glob
import json

texts = []
for filepath in glob.glob('source_data/*.txt'):
    with open(filepath, 'r') as f:
        texts.append({"text": f.read()})

with open('data/train.jsonl', 'w') as f:
    for obj in texts:
        f.write(json.dumps(obj) + '\n')
```

## Using Hugging Face Datasets

You don't need to download - just use the dataset name:

```bash
python scripts/train.py \
    --dataset_path "wikitext" \
    --text_field "text" \
    --split "train"
```

Popular datasets:
- `wikitext` - Wikipedia articles
- `openwebtext` - Web text
- `bookcorpus` - Books
- `c4` - Common Crawl

## Data Structure After Setup

```
data/
├── train.txt              # Your training data
├── eval.txt               # Optional: evaluation data
├── sample_train.jsonl     # Generated sample (if using --create_sample_data)
└── README.md              # This file
```

## Next Steps

Once you have data in this directory:

1. Verify format:
   ```bash
   head -n 5 data/train.txt
   ```

2. Check size:
   ```bash
   wc -l data/train.txt
   ```

3. Start training:
   ```bash
   python scripts/train.py --dataset_path data/train.txt
   ```

## Troubleshooting

**Issue**: "Dataset not found"
- Ensure file is in `data/` directory
- Check filename matches `--dataset_path` argument

**Issue**: "Cannot find 'text' field"
- For JSON/JSONL: Ensure you have a "text" field
- Or specify custom field: `--text_field your_field_name`

**Issue**: "Empty dataset"
- Check file is not empty: `cat data/train.txt`
- Verify format is correct

**Issue**: "Out of memory during data loading"
- Reduce `--max_length` (e.g., to 256 or 512)
- Use smaller batch size
- Process data in chunks

For more help, see the main [README.md](../README.md) and [SETUP.md](../SETUP.md).

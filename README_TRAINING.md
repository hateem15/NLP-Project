# Roman Urdu Sentiment Analysis and NER Training Guide

This guide explains how to train sentiment analysis and Named Entity Recognition (NER) models for Roman Urdu using Flair.

## Dataset Format

### Sentiment Analysis Dataset (`sentiment.csv`)
- Format: CSV with header row
- Columns:
  - Column 0: `sentence` - Roman Urdu text
  - Column 1: `sentiment` - Label (Positive/Negative)

### NER Dataset (`ner.txt`)
- Format: BIO tagging format
- Structure:
  - Each line contains: `word NER_TAG`
  - Empty lines separate sentences
  - Tags: `B-LOC`, `I-LOC`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-DATE`, `I-DATE`, `O`

## Quick Start

### Option 1: Train Both Models (Recommended)

Run the main script that handles everything:

```bash
python train_all.py
```

This will:
1. Split your datasets into train/dev/test splits
2. Train the sentiment analysis model
3. Train the NER model

### Option 2: Train Models Separately

#### Step 1: Prepare Data Splits

```bash
python prepare_data.py
```

This creates:
- `data/sentiment/train.csv`, `dev.csv`, `test.csv`
- `data/ner/train.txt`, `dev.txt`, `test.txt`

#### Step 2: Train Sentiment Model

```bash
python train_sentiment.py
```

Model will be saved to: `resources/taggers/roman-urdu-sentiment/`

#### Step 3: Train NER Model

```bash
python train_ner.py
```

Model will be saved to: `resources/taggers/roman-urdu-ner/`

## Using Trained Models

After training, you can use the models for inference:

```bash
python inference_example.py
```

Or use them in your own code:

```python
from flair.data import Sentence
from flair.models import TextClassifier, SequenceTagger

# Load sentiment model
sentiment_model = TextClassifier.load('resources/taggers/roman-urdu-sentiment/final-model.pt')

# Predict sentiment
sentence = Sentence("sahi bt h")
sentiment_model.predict(sentence)
print(sentence.labels[0].value)  # Positive or Negative

# Load NER model
ner_model = SequenceTagger.load('resources/taggers/roman-urdu-ner/final-model.pt')

# Predict entities
sentence = Sentence("Asad ne Telenor office mein join kiya")
ner_model.predict(sentence)
print(sentence.to_tagged_string('ner'))
```

## Model Configuration

### Sentiment Analysis Model
- **Embeddings**: XLM-RoBERTa-base (multilingual transformer)
- **Architecture**: TextClassifier with fine-tuned transformer
- **Learning Rate**: 5e-5
- **Batch Size**: 16
- **Epochs**: 10

### NER Model
- **Embeddings**: XLM-RoBERTa-base (multilingual transformer)
- **Architecture**: SequenceTagger with CRF layer
- **Learning Rate**: 5e-5
- **Batch Size**: 16
- **Epochs**: 10
- **Tag Format**: BIO

## Customization

You can modify the training parameters in the respective training scripts:

- **Learning Rate**: Change `learning_rate` parameter
- **Batch Size**: Change `mini_batch_size` parameter
- **Epochs**: Change `max_epochs` parameter
- **Model**: Change the transformer model (e.g., `'distilbert-base-multilingual-cased'`)

## Requirements

Make sure you have Flair installed:

```bash
pip install flair
```

Or install from the repository:

```bash
pip install -e .
```

## Troubleshooting

1. **Out of Memory**: Reduce `mini_batch_size` in the training scripts
2. **Slow Training**: Use a smaller transformer model like `'distilbert-base-multilingual-cased'`
3. **Model Not Found**: Make sure you've trained the models first using the training scripts

## Output

After training, you'll find:
- Trained models in `resources/taggers/`
- Training logs showing loss and metrics
- Model cards with training information

## Notes

- The scripts use a fixed random seed (42) for reproducibility
- Data splits are: 80% train, 10% dev, 10% test
- Models use multilingual transformers suitable for Roman Urdu
- Training may take some time depending on your hardware and dataset size


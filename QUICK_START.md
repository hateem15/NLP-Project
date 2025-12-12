# Quick Start Guide - Roman Urdu Sentiment Analysis & NER

## Overview

This repository contains scripts to train Flair models for:
1. **Sentiment Analysis** - Classify Roman Urdu text as Positive or Negative
2. **Named Entity Recognition (NER)** - Identify entities (Person, Location, Organization, Date) in Roman Urdu text

## Files Created

- `prepare_data.py` - Splits datasets into train/dev/test
- `train_sentiment.py` - Trains sentiment analysis model
- `train_ner.py` - Trains NER model  
- `train_all.py` - Main script that runs everything
- `inference_example.py` - Example usage of trained models
- `README_TRAINING.md` - Detailed documentation

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install flair
```

Or if installing from source:
```bash
pip install -e .
```

### Step 2: Run Training

Simply run:
```bash
python train_all.py
```

This will:
- ✅ Split your `sentiment.csv` and `ner.txt` into train/dev/test
- ✅ Train sentiment analysis model
- ✅ Train NER model

### Step 3: Use Models

```bash
python inference_example.py
```

## Dataset Requirements

Your datasets should be in the root directory:
- `sentiment.csv` - CSV with columns: `sentence`, `sentiment`
- `ner.txt` - BIO format with word and tag per line

## Model Outputs

After training, models are saved to:
- Sentiment: `resources/taggers/roman-urdu-sentiment/final-model.pt`
- NER: `resources/taggers/roman-urdu-ner/final-model.pt`

## Customization

Edit the training scripts to adjust:
- Learning rate
- Batch size
- Number of epochs
- Transformer model (currently using `xlm-roberta-base`)

## Troubleshooting

**Out of memory?** 
- Reduce `mini_batch_size` in training scripts (try 8 or 4)

**Want faster training?**
- Use `'distilbert-base-multilingual-cased'` instead of `'xlm-roberta-base'`

**Need better accuracy?**
- Increase `max_epochs`
- Use larger transformer models
- For NER: Set `use_context=True` in embeddings (FLERT approach)

## Example Usage

```python
from flair.data import Sentence
from flair.models import TextClassifier, SequenceTagger

# Sentiment
sentiment_model = TextClassifier.load('resources/taggers/roman-urdu-sentiment/final-model.pt')
sentence = Sentence("sahi bt h")
sentiment_model.predict(sentence)
print(sentence.labels[0].value)  # Positive/Negative

# NER
ner_model = SequenceTagger.load('resources/taggers/roman-urdu-ner/final-model.pt')
sentence = Sentence("Asad ne Telenor office mein join kiya")
ner_model.predict(sentence)
print(sentence.to_tagged_string('ner'))
```

## Next Steps

1. Run `python train_all.py` to train both models
2. Check `README_TRAINING.md` for detailed documentation
3. Use `inference_example.py` to see how to use the models
4. Integrate models into your application!


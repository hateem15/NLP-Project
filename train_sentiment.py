"""
Train a sentiment analysis model for Roman Urdu using Flair
"""
import logging
from pathlib import Path

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flair")


def train_sentiment_model():
    """
    Train a sentiment analysis model for Roman Urdu
    """
    # 1. Define data folder and column mapping
    data_folder = Path('data/sentiment')
    
    # Column mapping: column 0 is text, column 1 is label
    column_name_map = {0: "text", 1: "label"}
    
    # 2. Load the corpus
    logger.info("Loading sentiment corpus...")
    corpus: Corpus = CSVClassificationCorpus(
        data_folder,
        column_name_map,
        label_type='sentiment',
        skip_header=True,  # Skip CSV header
        encoding='utf-8',
        in_memory=True
    )
    
    logger.info(f"Corpus loaded: {corpus}")
    logger.info(f"Train: {len(corpus.train)} samples")
    logger.info(f"Dev: {len(corpus.dev)} samples")
    logger.info(f"Test: {len(corpus.test)} samples")
    
    # 3. Create label dictionary
    label_type = 'sentiment'
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    logger.info(f"Label dictionary: {label_dict}")
    
    # 4. Initialize transformer document embeddings
    # Using multilingual model for Roman Urdu
    # Options: 'xlm-roberta-base', 'distilbert-base-multilingual-cased', 'bert-base-multilingual-cased'
    logger.info("Initializing transformer embeddings...")
    document_embeddings = TransformerDocumentEmbeddings(
        'xlm-roberta-base',  # Good multilingual model
        fine_tune=True
    )
    
    # 5. Create the text classifier
    logger.info("Creating text classifier...")
    classifier = TextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        label_type=label_type
    )
    
    # 6. Initialize trainer
    trainer = ModelTrainer(classifier, corpus)
    
    # 7. Run training with fine-tuning
    logger.info("Starting training...")
    trainer.fine_tune(
        'resources/taggers/roman-urdu-sentiment',
        learning_rate=5.0e-5,
        mini_batch_size=16,
        max_epochs=4,
        train_with_dev=False,
        monitor_test=True,
        monitor_train_sample=0.1,  # Monitor 10% of training data
    )
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: resources/taggers/roman-urdu-sentiment")
    
    # Print model card
    classifier.print_model_card()


if __name__ == '__main__':
    train_sentiment_model()


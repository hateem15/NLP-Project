"""
Train a Named Entity Recognition (NER) model for Roman Urdu using Flair
"""
import logging
from pathlib import Path

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flair")


def train_ner_model():
    """
    Train a NER model for Roman Urdu
    """
    # 1. Define data folder and column format
    data_folder = Path('data/ner')
    
    # Column format: column 0 is text (word), column 1 is NER tag
    columns = {0: 'text', 1: 'ner'}
    
    # 2. Load the corpus
    logger.info("Loading NER corpus...")
    corpus: Corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file='train.txt',
        dev_file='dev.txt',
        test_file='test.txt',
        encoding='utf-8',
        in_memory=True
    )
    
    logger.info(f"Corpus loaded: {corpus}")
    logger.info(f"Train: {len(corpus.train)} sentences")
    logger.info(f"Dev: {len(corpus.dev)} sentences")
    logger.info(f"Test: {len(corpus.test)} sentences")
    
    # Print a sample sentence
    if len(corpus.train) > 0:
        logger.info(f"Sample sentence: {corpus.train[0].to_tagged_string('ner')}")
    
    # 3. Create label dictionary
    label_type = 'ner'
    tag_dictionary = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    logger.info(f"Tag dictionary: {tag_dictionary}")
    logger.info(f"Number of tags: {len(tag_dictionary)}")
    
    # 4. Initialize transformer word embeddings
    # Using multilingual model for Roman Urdu
    logger.info("Initializing transformer embeddings...")
    embeddings = TransformerWordEmbeddings(
        model='xlm-roberta-base',  # Good multilingual model
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,  # Set to True for FLERT (better but slower)
    )
    
    # 5. Initialize sequence tagger
    logger.info("Creating sequence tagger...")
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=label_type,
        use_crf=True,  # CRF helps with sequence labeling
        use_rnn=False,  # Not needed with transformers
        reproject_embeddings=False,  # Not needed with transformers
    )
    
    # 6. Initialize trainer
    trainer = ModelTrainer(tagger, corpus)
    
    # 7. Run fine-tuning
    logger.info("Starting training...")
    trainer.fine_tune(
        'resources/taggers/roman-urdu-ner',
        learning_rate=5.0e-5,
        mini_batch_size=16,
        max_epochs=5,
        train_with_dev=False,
        monitor_test=True,
        monitor_train_sample=0.1,  # Monitor 10% of training data
    )
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: resources/taggers/roman-urdu-ner")
    
    # Print model card
    tagger.print_model_card()


if __name__ == '__main__':
    train_ner_model()


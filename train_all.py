"""
Main script to train both sentiment analysis and NER models for Roman Urdu
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to prepare data and train both models
    """
    logger.info("=" * 80)
    logger.info("Roman Urdu Sentiment Analysis and NER Training")
    logger.info("=" * 80)
    
    # Step 1: Prepare data
    logger.info("\n[Step 1/3] Preparing data splits...")
    try:
        from prepare_data import split_sentiment_data, split_ner_data
        
        if not Path('data/sentiment/train.csv').exists():
            logger.info("Splitting sentiment data...")
            split_sentiment_data()
        else:
            logger.info("Sentiment data splits already exist, skipping...")
        
        if not Path('data/ner/train.txt').exists():
            logger.info("Splitting NER data...")
            split_ner_data()
        else:
            logger.info("NER data splits already exist, skipping...")
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        sys.exit(1)
    
    # Step 2: Train sentiment model
    logger.info("\n[Step 2/3] Training sentiment analysis model...")
    try:
        from train_sentiment import train_sentiment_model
        train_sentiment_model()
        logger.info("✓ Sentiment model training completed successfully!")
    except Exception as e:
        logger.error(f"Error training sentiment model: {e}")
        import traceback
        traceback.print_exc()
        logger.warning("Continuing with NER training...")
    
    # Step 3: Train NER model
    logger.info("\n[Step 3/3] Training NER model...")
    try:
        from train_ner import train_ner_model
        train_ner_model()
        logger.info("✓ NER model training completed successfully!")
    except Exception as e:
        logger.error(f"Error training NER model: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline completed!")
    logger.info("=" * 80)
    logger.info("\nTrained models:")
    logger.info("  - Sentiment: resources/taggers/roman-urdu-sentiment")
    logger.info("  - NER: resources/taggers/roman-urdu-ner")
    logger.info("\nYou can now use these models for inference!")


if __name__ == '__main__':
    main()


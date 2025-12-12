"""
Interactive inference script for sentiment analysis and NER models
"""
from flair.data import Sentence
from flair.models import TextClassifier, SequenceTagger


def sentiment_inference(classifier, num_sentences):
    """
    Run sentiment analysis inference on user-provided sentences
    """
    print("\n" + "=" * 80)
    print("Sentiment Analysis Inference")
    print("=" * 80)
    
    sentences = []
    for i in range(num_sentences):
        text = input(f"\nEnter sentence {i+1}/{num_sentences}: ").strip()
        if text:
            sentences.append(text)
    
    print("\n" + "-" * 80)
    print("Results:")
    print("-" * 80)
    
    for text in sentences:
        sentence = Sentence(text)
        classifier.predict(sentence)
        
        label = sentence.labels[0].value
        confidence = sentence.labels[0].score
        
        print(f"\nText: {text}")
        print(f"Sentiment: {label} (confidence: {confidence:.4f})")


def ner_inference(tagger, num_sentences):
    """
    Run NER inference on user-provided sentences
    """
    print("\n" + "=" * 80)
    print("Named Entity Recognition Inference")
    print("=" * 80)
    
    sentences = []
    for i in range(num_sentences):
        text = input(f"\nEnter sentence {i+1}/{num_sentences}: ").strip()
        if text:
            sentences.append(text)
    
    print("\n" + "-" * 80)
    print("Results:")
    print("-" * 80)
    
    for text in sentences:
        sentence = Sentence(text)
        tagger.predict(sentence)
        
        print(f"\nText: {text}")
        print(f"Tagged: {sentence.to_tagged_string('ner')}")
        
        # Print entities
        entities = sentence.get_spans('ner')
        if entities:
            print("Entities found:")
            for entity in entities:
                print(f"  - {entity.text}: {entity.tag}")
        else:
            print("No entities found.")


def both_inference(classifier, tagger, num_sentences):
    """
    Run both sentiment and NER inference on user-provided sentences
    """
    print("\n" + "=" * 80)
    print("Sentiment Analysis & Named Entity Recognition Inference")
    print("=" * 80)
    
    sentences = []
    for i in range(num_sentences):
        text = input(f"\nEnter sentence {i+1}/{num_sentences}: ").strip()
        if text:
            sentences.append(text)
    
    print("\n" + "-" * 80)
    print("Results:")
    print("-" * 80)
    
    for text in sentences:
        sentence = Sentence(text)
        
        # Run both predictions
        classifier.predict(sentence)
        tagger.predict(sentence)
        
        # Get sentiment
        sentiment_label = sentence.labels[0].value
        sentiment_confidence = sentence.labels[0].score
        
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment_label} (confidence: {sentiment_confidence:.4f})")
        print(f"Tagged: {sentence.to_tagged_string('ner')}")
        
        # Print entities
        entities = sentence.get_spans('ner')
        if entities:
            print("Entities found:")
            for entity in entities:
                print(f"  - {entity.text}: {entity.tag}")
        else:
            print("No entities found.")


def main():
    """
    Main interactive function
    """
    print("=" * 80)
    print("Roman Urdu Sentiment Analysis & NER Inference")
    print("=" * 80)
    print("\nChoose an option:")
    print("  1. Sentiment Analysis only")
    print("  2. Named Entity Recognition (NER) only")
    print("  3. Both Sentiment Analysis and NER")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice! Please enter 1, 2, or 3.")
    
    # Ask for number of sentences
    while True:
        try:
            num_sentences = int(input("\nHow many test sentences do you want to provide? "))
            if num_sentences > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load models based on choice
    classifier = None
    tagger = None
    
    if choice == '1':
        # Load sentiment model
        try:
            print("\nLoading sentiment model...")
            classifier = TextClassifier.load('resources/taggers/roman-urdu-sentiment/final-model.pt')
            print("✓ Sentiment model loaded successfully!")
        except FileNotFoundError:
            print("Error: Sentiment model not found!")
            print("Please train the model first using train_sentiment.py")
            return
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            return
        
        # Run sentiment inference
        sentiment_inference(classifier, num_sentences)
    
    elif choice == '2':
        # Load NER model
        try:
            print("\nLoading NER model...")
            tagger = SequenceTagger.load('resources/taggers/roman-urdu-ner/final-model.pt')
            print("✓ NER model loaded successfully!")
        except FileNotFoundError:
            print("Error: NER model not found!")
            print("Please train the model first using train_ner.py")
            return
        except Exception as e:
            print(f"Error loading NER model: {e}")
            return
        
        # Run NER inference
        ner_inference(tagger, num_sentences)
    
    elif choice == '3':
        # Load both models
        try:
            print("\nLoading sentiment model...")
            classifier = TextClassifier.load('resources/taggers/roman-urdu-sentiment/final-model.pt')
            print("✓ Sentiment model loaded successfully!")
        except FileNotFoundError:
            print("Error: Sentiment model not found!")
            print("Please train the model first using train_sentiment.py")
            return
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            return
        
        try:
            print("Loading NER model...")
            tagger = SequenceTagger.load('resources/taggers/roman-urdu-ner/final-model.pt')
            print("✓ NER model loaded successfully!")
        except FileNotFoundError:
            print("Error: NER model not found!")
            print("Please train the model first using train_ner.py")
            return
        except Exception as e:
            print(f"Error loading NER model: {e}")
            return
        
        # Run both inferences
        both_inference(classifier, tagger, num_sentences)
    
    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

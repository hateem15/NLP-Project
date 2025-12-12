"""
Data preparation script to split datasets into train/dev/test splits
"""
import csv
import os
import random
from pathlib import Path

def split_sentiment_data(input_file='sentiment.csv', output_dir='data/sentiment', train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """
    Split sentiment CSV file into train, dev, and test sets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all data
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(reader)  # Skip header
        for row in reader:
            # Skip rows that don't have exactly 2 columns (might be malformed)
            if len(row) >= 2:
                rows.append(row)
            else:
                print(f"Warning: Skipping malformed row: {row}")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(rows)
    
    # Calculate split indices
    total = len(rows)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    # Split data
    train_data = rows[:train_end]
    dev_data = rows[train_end:dev_end]
    test_data = rows[dev_end:]
    
    # Write splits
    for split_name, split_data in [('train', train_data), ('dev', dev_data), ('test', test_data)]:
        output_file = output_dir / f'{split_name}.csv'
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write header
            writer.writerows(split_data)
        print(f"Created {output_file} with {len(split_data)} samples")
    
    print(f"\nSentiment data split complete:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Dev: {len(dev_data)} samples")
    print(f"  Test: {len(test_data)} samples")


def split_ner_data(input_file='ner.txt', output_dir='data/ner', train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """
    Split NER BIO format file into train, dev, and test sets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all sentences (separated by empty lines)
    sentences = []
    current_sentence = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        
        # Add last sentence if file doesn't end with empty line
        if current_sentence:
            sentences.append(current_sentence)
    
    # Shuffle sentences
    random.seed(42)
    random.shuffle(sentences)
    
    # Calculate split indices
    total = len(sentences)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    # Split data
    train_sentences = sentences[:train_end]
    dev_sentences = sentences[train_end:dev_end]
    test_sentences = sentences[dev_end:]
    
    # Write splits
    for split_name, split_sentences in [('train', train_sentences), ('dev', dev_sentences), ('test', test_sentences)]:
        output_file = output_dir / f'{split_name}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in split_sentences:
                for line in sentence:
                    f.write(line + '\n')
                f.write('\n')  # Empty line between sentences
        print(f"Created {output_file} with {len(split_sentences)} sentences")
    
    print(f"\nNER data split complete:")
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Dev: {len(dev_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")


if __name__ == '__main__':
    print("Preparing sentiment analysis data...")
    split_sentiment_data()
    
    print("\nPreparing NER data...")
    split_ner_data()
    
    print("\nData preparation complete!")


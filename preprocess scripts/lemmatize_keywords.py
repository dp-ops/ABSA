import json
import spacy
import os

# Load the Greek spaCy model
# Ensure you have this model downloaded: python -m spacy download el_core_news_lg
try:
    nlp = spacy.load("el_core_news_lg")
except OSError:
    print("Downloading el_core_news_lg model...")
    spacy.cli.download("el_core_news_lg")
    nlp = spacy.load("el_core_news_lg")

def lemmatize_word(word_text):
    """Lemmatizes a single word or a short phrase."""
    # spaCy's nlp() is best for processing sentences or documents.
    # For single words or short phrases that are keywords,
    # we process them as a small document.
    doc = nlp(str(word_text).lower()) # Lowercase before lemmatization for consistency
    # We join the lemmas of the tokens. If it's a single word, it will be a single lemma.
    # If it's a phrase, it will be a lemmatized phrase.
    return " ".join([token.lemma_ for token in doc])

def main():
    # Define paths relative to the workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Go up one level from 'preprocess scripts'
    input_keyword_file = os.path.join(workspace_root, 'data', 'aspect_keywords_map.json')
    output_keyword_file = os.path.join(os.path.dirname(__file__), 'aspect_keywords_lemma.json') # Save in the same folder as this script

    # Load the original aspect keywords
    try:
        with open(input_keyword_file, 'r', encoding='utf-8') as f:
            aspect_keywords_map = json.load(f)
        print(f"Successfully loaded keywords from {input_keyword_file}")
    except FileNotFoundError:
        print(f"Error: Input keyword file not found at {input_keyword_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_keyword_file}")
        return

    lemmatized_aspect_map = {}

    print("Starting lemmatization process...")
    for aspect, keywords in aspect_keywords_map.items():
        lemmatized_keywords = set() # Use a set to automatically handle duplicates
        for keyword in keywords:
            # Skip non-Greek or very short keywords if necessary, or purely symbolic ones
            # For now, we attempt to lemmatize all, assuming lemmatizer handles various inputs gracefully.
            # English words like 'camera', 'call' will likely remain unchanged by a Greek lemmatizer.
            lemmatized_keyword = lemmatize_word(keyword)
            lemmatized_keywords.add(lemmatized_keyword)
        
        lemmatized_aspect_map[aspect] = sorted(list(lemmatized_keywords)) # Convert back to sorted list for consistent output
        print(f"Processed aspect: {aspect} - Original keywords: {len(keywords)}, Lemmatized unique keywords: {len(lemmatized_aspect_map[aspect])}")

    # Save the lemmatized keywords
    try:
        with open(output_keyword_file, 'w', encoding='utf-8') as f:
            json.dump(lemmatized_aspect_map, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved lemmatized keywords to {output_keyword_file}")
    except IOError:
        print(f"Error: Could not write to output file at {output_keyword_file}")

if __name__ == "__main__":
    main() 
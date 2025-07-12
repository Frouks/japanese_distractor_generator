import pickle
from pathlib import Path
import time
import random

# --- Configuration ---
TRIGRAM_FILE = Path("processed_corpus_data") / "jp_trigram_counts.pkl"

def load_data(file_path):
    """Loads the pruned trigram set from the pickle file."""
    if not file_path.exists():
        print(f"❌ ERROR: File not found at {file_path}")
        return None

    print(f"Loading data from {file_path}... (This may take a moment)")
    start_time = time.time()
    try:
        with open(file_path, 'rb') as f:
            trigram_set = pickle.load(f)
        
        duration = time.time() - start_time
        print(f"✅ Data loaded successfully in {duration:.2f} seconds.")
        print(f"Found {len(trigram_set):,} unique trigrams (with count >= 5).")
        return trigram_set

    except Exception as e:
        print(f"❌ An error occurred during loading: {e}")
        return None

def show_random_n(trigram_set, n=20):
    """Displays N random items from the set."""
    if not trigram_set:
        return
    print(f"\n--- Showing {n} Random Trigrams ---")
    
    # Convert set to list to be able to pick random elements
    trigram_list = list(trigram_set)
    if len(trigram_list) < n:
        print("Not enough trigrams to show the requested number.")
        n = len(trigram_list)
        
    random_items = random.sample(trigram_list, n)
    for w1, w2, w3 in random_items:
        print(f"  ('{w1}', '{w2}', '{w3}')")

def search_for_word(trigram_set, search_word, limit=50):
    """Finds and displays all trigrams containing a specific word."""
    if not trigram_set:
        return
    print(f"\n--- Searching for trigrams containing the word: '{search_word}' ---")
    
    # This list comprehension will find all relevant trigrams
    found_trigrams = [
        trigram for trigram in trigram_set if search_word in trigram
    ]
    
    if not found_trigrams:
        print(f"No trigrams found containing '{search_word}'.")
        return
        
    print(f"Found {len(found_trigrams):,} trigrams. Showing a random sample of up to {limit}:")
    
    # Shuffle and show a sample to avoid alphabetical bias
    random.shuffle(found_trigrams)
    for w1, w2, w3 in found_trigrams[:limit]:
        print(f"  ('{w1}', '{w2}', '{w3}')")

if __name__ == "__main__":
    trigram_data = load_data(TRIGRAM_FILE)

    if trigram_data:
        # --- Example Usage ---
        
        # 1. Show some random trigrams to get a feel for the data
        show_random_n(trigram_data, n=25)
        
        # 2. Search for trigrams involving specific words of interest.
        search_for_word(trigram_data, "猫") # Cat
        
        search_for_word(trigram_data, "東京") # Tokyo

        search_for_word(trigram_data, "走る") # run

        search_for_word(trigram_data, "美しい") # beautiful

        search_for_word(trigram_data, "の") # the particle 'no'
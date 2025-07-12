import pickle
from pathlib import Path
import time
import random

# --- Configuration ---
# DEPENDENCY_FILE = Path("processed_corpus_data") / "jp_dependency_relations_SAMPLE.pkl"
DEPENDENCY_FILE = Path("processed_corpus_data") / "jp_dependency_relations.pkl"

def load_data(file_path):
    """Loads the dependency inverted index from the pickle file."""
    if not file_path.exists():
        print(f"❌ ERROR: File not found at {file_path}")
        return None

    print(f"Loading data from {file_path}...")
    start_time = time.time()
    try:
        with open(file_path, 'rb') as f:
            # The file now contains a dictionary (our inverted index)
            dependency_index = pickle.load(f)
        
        duration = time.time() - start_time
        print(f"✅ Data loaded successfully in {duration:.2f} seconds.")
        # The length of the dict is the number of unique word keys
        print(f"Found an index with {len(dependency_index):,} unique word keys.")
        return dependency_index

    except Exception as e:
        print(f"❌ An error occurred during loading: {e}")
        return None

def show_random_n(dependency_index, n=20):
    """Displays N random relations from the index."""
    if not dependency_index:
        return
    print(f"\n--- Showing {n} Random Dependency Relations from the Index ---")
    
    # Get a random sample of keys (words) from the index
    all_keys = list(dependency_index.keys())
    if not all_keys:
        print("Index is empty.")
        return
        
    # Pick a few random words and show one of their relations
    for _ in range(n):
        random_word = random.choice(all_keys)
        # Get the list of relations for that word
        relations_for_word = dependency_index[random_word]
        if relations_for_word:
            # Pick a random relation from that list
            random_relation = random.choice(relations_for_word)
            dep, head, child = random_relation
            print(f"  ({dep}) : {head} <--- {child}  (from key: '{random_word}')")

def search_for_word(dependency_index, search_word, limit=50):
    """Finds and displays relations for a specific word using the index."""
    if not dependency_index:
        return
    print(f"\n--- Searching for relations involving the lemma: '{search_word}' ---")
    
    # This is now an O(1) lightning-fast lookup!
    found_relations = dependency_index.get(search_word, [])
    
    if not found_relations:
        print(f"No relations found for key '{search_word}'.")
        return
        
    print(f"Found {len(found_relations)} relations for '{search_word}'. Showing a random sample of up to {limit}:")
    
    # Shuffle the list of found relations to get a random sample
    random.shuffle(found_relations)
    for dep, head, child in found_relations[:limit]:
        if head == search_word:
            print(f"  ({dep}) : [{head}] <--- {child}")
        else:
            print(f"  ({dep}) : {head} <--- [{child}]")

if __name__ == "__main__":
    # Load the inverted index data
    dependency_data_index = load_data(DEPENDENCY_FILE)

    if dependency_data_index:
        # --- Example Usage ---
        show_random_n(dependency_data_index, n=25)
        
        search_for_word(dependency_data_index, "食べる")
        search_for_word(dependency_data_index, "猫")
        search_for_word(dependency_data_index, "美しい")
        search_for_word(dependency_data_index, "書く")
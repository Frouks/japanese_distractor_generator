# ========================================================================================
#
# PURPOSE:
# Generates distractors that are semantically similar to the target word. This is
# achieved by finding words that are "close" to the target word in a high-dimensional
# vector space.
#
# ========================================================================================
#
#
# This generator is a significant improvement over the previous system's Word2Vec-based
# generator for several key reasons:
#
# 1.  STATE-OF-THE-ART PRE-TRAINED MODEL:
#     We are using a FastText model that has been pre-trained by Facebook AI on a massive
#     Japanese text corpus (Common Crawl). This saves us the significant time and
#     computational cost of training our own model and provides world-class, general-
#     purpose word representations out of the box.
#
# 2.  SUPERIOR HANDLING OF UNKNOWN WORDS (OOV - Out-of-Vocabulary):
#     - OLD WAY (Word2Vec): Could only generate vectors for words it had seen during
#       training. If you provided a new slang term, a technical word, or a rare name,
#       the model would fail completely.
#     - NEW WAY (FastText): Learns vectors for character n-grams (subword units).
#       For an unknown word like the modern slang `デジ変` (dejiben), it can construct
#       a meaningful vector by combining the vectors of its known parts like `デジ`
#       and `ジ変`. This makes the system incredibly robust for real-world text.
#
# 3.  HIGHER QUALITY VECTORS FOR RARE WORDS:
#     Even if a word is rare, its subword parts (e.g., the `可愛い` in the made-up
#     word `怖可愛い`) are often common. By building a word's vector from these common
#     parts, FastText produces much more accurate and nuanced representations for
#     infrequent words compared to Word2Vec.
#
# ========================================================================================

from gensim.models.fasttext import load_facebook_model

class SimilarityGenerator:
    """
    Generates distractors that are semantically similar to the target word
    using a pre-trained FastText model.
    """
    def __init__(self, model_path: str):
        """
        Initializes the generator by loading the FastText model. This is a one-time
        cost when the system starts up.

        Args:
            model_path (str): The path to the downloaded FastText model file.
                              The model is expected to be in the word2vec text format.
        """
        print("Initializing SimilarityGenerator...")
        # The model is loaded into a KeyedVectors object, which is a memory-efficient
        # data structure designed specifically for holding and querying word vectors.
        # This process can take a few moments and may consume a significant amount of RAM.
        try:
            # We use `load_word2vec_format` which is compatible with the standard
            # text format provided by FastText.
            self.model = load_facebook_model(model_path).wv
            print(f"✅ FastText model loaded successfully from '{model_path}'")
        except FileNotFoundError:
            print(f"❌ MODEL FILE NOT FOUND at '{model_path}'.")
            print("Please download the pre-trained Japanese FastText model and place it in your project directory.")
            print("You can download it from: https://fasttext.cc/docs/en/crawl-vectors.html")
            self.model = None

    def generate(self, target_word: str, top_n: int = 20) -> list[tuple[str, float]]:
        """
        Finds the most similar words to the target word in the vector space.

        Args:
            target_word (str): The word for which to find similar distractors.
            top_n (int): The number of similar distractors to return.

        Returns:
            list[tuple[str, float]]: A list of (word, similarity_score) tuples,
                                     ranked from most to least similar. Returns an
                                     empty list if the model isn't loaded.
        """
        # First, we check if the model was loaded successfully during initialization.
        if self.model is None:
            print("Error: Model not loaded. Cannot generate similarity distractors.")
            return []

        try:
            # This is the core function call. The `most_similar` method efficiently
            # calculates the cosine similarity between the vector for the `target_word`
            # and all other word vectors in the model's vocabulary. It returns the
            # `top_n` words with the highest similarity scores.
            similar_words = self.model.most_similar(target_word, topn=top_n)
            print(f"Successfully generated {len(similar_words)} similarity distractors for '{target_word}'.")
            return similar_words
        except KeyError:
            # This 'except' block is less likely to be hit with FastText compared to
            # Word2Vec, but it's good practice to include it. It handles cases where
            # a word is so unusual that even its subword parts are not recognized.
            print(f"Warning: Could not generate a vector for the target word '{target_word}'.")
            return []

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # IMPORTANT: You must download the pre-trained Japanese FastText model file first.
    # It is typically named 'cc.ja.300.vec.gz'. Unzip it before use.
    # Download from: https://fasttext.cc/docs/en/crawl-vectors.html
    FASTTEXT_MODEL_PATH = '../models/cc.ja.300.bin'

    # --- INITIALIZATION ---
    # Create an instance of the generator.
    sim_generator = SimilarityGenerator(model_path=FASTTEXT_MODEL_PATH)
    print("-" * 50)

    # Proceed with tests only if the model was loaded successfully.
    if sim_generator.model:
        # --- TEST CASES ---

        # Test Case 1: A common noun, "cat"
        target_1 = "猫"
        distractors_1 = sim_generator.generate(target_1)
        print("\nTop 10 distractors for '猫' (cat):")
        # We slice the list `[:5]` to only show the top 5 results for clarity.
        for word, score in distractors_1[:10]:
            # The formatting `<10` aligns the text for better readability.
            print(f"  - {word:<10} (Similarity: {score:.4f})")
        print("-" * 50)

        # Test Case 2: A common verb, "to run"
        target_2 = "走る"
        distractors_2 = sim_generator.generate(target_2)
        print("\nTop 10 distractors for '走る' (run):")
        for word, score in distractors_2[:10]:
            print(f"  - {word:<10} (Similarity: {score:.4f})")
        print("-" * 50)

        # Test Case 3: A more abstract concept, "government"
        target_3 = "政府"
        distractors_3 = sim_generator.generate(target_3)
        print("\nTop 105 distractors for '政府' (government):")
        for word, score in distractors_3[:10]:
            print(f"  - {word:<10} (Similarity: {score:.4f})")
        print("-" * 50)

        # Test Case 4: An English loanword, "computer"
        target_4 = "コンピュータ"
        distractors_4 = sim_generator.generate(target_4)
        print("\nTop 10 distractors for 'コンピュータ' (computer):")
        for word, score in distractors_4[:10]:
            print(f"  - {word:<10} (Similarity: {score:.4f})")
        print("-" * 50)
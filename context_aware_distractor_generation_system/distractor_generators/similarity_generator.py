import logging
import sys
from pathlib import Path
from typing import Union

from gensim.models.fasttext import load_facebook_model

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from context_aware_distractor_generation_system.constants.SentenceContextEnum import SentenceContextEnum


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
        """
        print("Initializing SimilarityGenerator...")
        try:
            self.model = load_facebook_model(model_path).wv
            self.logger = logging.getLogger('SimilarityGenerator')
            print(f"✅ FastText model loaded successfully from '{model_path}'")
        except FileNotFoundError:
            print(f"❌ MODEL FILE NOT FOUND at '{model_path}'.")
            self.model = None

    # def generate_distractors(self, target_word: str, top_n: int = 20) -> list[tuple[str, float]]:
    #     """
    #     Finds the most similar words to the target word in the vector space.
    #
    #     Args:
    #         target_word (str): The word for which to find similar distractors.
    #         top_n (int): The number of similar distractors to return.
    #
    #     Returns:
    #         list[tuple[str, float]]: A list of (word, similarity_score) tuples,
    #                                  ranked from most to least similar. Returns an
    #                                  empty list if the model isn't loaded.
    #     """
    #     if self.model is None:
    #         print("❌ Error: Model not loaded. Cannot generate similarity distractors.")
    #         return []
    #
    #     try:
    #         similar_words = self.model.most_similar(target_word, topn=top_n)
    #         print(f"Successfully generated {len(similar_words)} similarity distractors for '{target_word}'.")
    #         return similar_words
    #     except KeyError:
    #         # Less likely than for Word2Vec
    #         print(f"Warning: Could not generate a vector for the target word '{target_word}'.")
    #         return []

    def generate_distractors(
            self,
            target_word: str,
            context_type: SentenceContextEnum,
            top_n: int = 5,
            num_candidates: int = 200,
            include_sim_score: bool = True,
    ) -> Union[list[tuple[str, float]], list[str]]:
        """
            Finds similar words to the target word and filters them based on the
            sentence context (Open or Closed).

            Args:
                target_word (str): The word for which to find similar distractors.
                context_type (str): The type of context, 'Open' or 'Closed'.
                top_n (int): The final number of distractors to return.
                num_candidates (int): The number of initial candidates to fetch before filtering.
                include_sim_score (bool): If False, returns only a list of words.
                                        If True, returns a list of (word, score) tuples.

            Returns:
                A list of distractors in the specified format.
        """
        if self.model is None:
            print("Error: Model not loaded. Cannot generate similarity distractors.")
            return []

        # 1. Define similarity thresholds based on context
        if context_type == SentenceContextEnum.OPEN:
            # For open contexts, we want distractors that are related but not too close.
            min_similarity, max_similarity = 0.4, 0.6
        elif context_type == SentenceContextEnum.CLOSED:
            # For closed contexts, we want very similar distractors to be more challenging.
            min_similarity, max_similarity = 0.5, 0.70
        else:
            print(f"Warning: Invalid context_type '{context_type}'. Defaulting to a wide range.")
            min_similarity, max_similarity = 0.4, 0.75

        self.logger.info(
            f"--- Generating for '{target_word}' (Context: {context_type}, Range: {min_similarity}-{max_similarity}) ---")
        try:
            # 2. Fetch a large pool of candidates
            all_similar_words = self.model.most_similar(target_word, topn=num_candidates)

            # 3. Filter candidates based on the similarity range
            filtered_distractors = [
                (word, score)
                for word, score in all_similar_words
                if min_similarity <= score <= max_similarity
            ]

            # 4. Get the top N distractors from the filtered list
            final_distractors_with_scores = filtered_distractors[:top_n]
            print(f"Successfully generated {len(final_distractors_with_scores)} distractors after filtering.")

            # 5. Return the list in the desired format based on the flag.
            if include_sim_score:
                return final_distractors_with_scores
            else:
                return [word for word, score in final_distractors_with_scores]
        except KeyError:
            print(f"Warning: Could not generate a vector for the target word '{target_word}'.")
            return []


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.getLogger('gensim.models._fasttext_bin').setLevel(logging.CRITICAL)

    FASTTEXT_MODEL_PATH = '../model/cc.ja.300.bin'
    sim_generator = SimilarityGenerator(model_path=FASTTEXT_MODEL_PATH)
    print("=" * 60)

    if sim_generator.model:
        test_cases: list[dict[str, any]] = [
            {
                "sentence": "動物園で、大きな＿＿が鼻を高く上げていた。",
                "target": "象",
                "context": SentenceContextEnum.CLOSED,
                "english": "At the zoo, the big ___ was raising its trunk high."
            },
            {
                "sentence": "公園で、たくさんの＿＿が遊んでいた。",
                "target": "子供",
                "context": SentenceContextEnum.OPEN,
                "english": "At the park, many ___ were playing."
            },
            {
                "sentence": "彼は100メートルを10秒で＿＿ことができる。",
                "target": "走る",
                "context": SentenceContextEnum.CLOSED,
                "english": "He can ___ 100 meters in 10 seconds."
            },
            {
                "sentence": "この＿＿はとても重要です。",
                "target": "問題",
                "context": SentenceContextEnum.OPEN,
                "english": "This ___ is very important."
            }
        ]

        for i, case in enumerate(test_cases):
            context = "Open" if case["context"] == SentenceContextEnum.OPEN else "Closed"
            print(f"🧪 TEST CASE {i + 1}: {context} CONTEXT")
            print(f"   Sentence: {case['sentence']}")
            print(f"   English:  {case['english']}")
            print(f"   Target:   {case['target']}")
            print("-" * 60)

            distractors = sim_generator.generate_distractors(
                target_word=case['target'],
                context_type=case['context'],
                top_n=5,
                num_candidates=20,
                include_sim_score=True
            )

            print("\n   Generated Distractors:")
            if distractors:
                for word, score in distractors:
                    # Using <15 for alignment to handle wider characters
                    print(f"     - {word:<15} (Similarity: {score:.4f})")
            else:
                print("     No distractors found in the specified range.")
            print("=" * 60)

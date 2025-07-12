import logging
from gensim.models import Word2Vec
from pathlib import Path

class SimilarityGenerator:
    """
    Generates distractors based on semantic similarity using a pre-trained Word2Vec model.
    """
    def __init__(self, model_path, corpus_processor):
        """
        Initializes the SimilarityGenerator.

        Args:
            model_path (str or Path): Path to the saved Word2Vec model.
            corpus_processor (CorpusProcessor): An instance to analyze the target word.
        """
        self.logger = logging.getLogger('SimilarityGenerator')
        self.processor = corpus_processor
        self.model_path = Path(model_path)
        self.model = self._load_model()
        if self.model:
            self.logger.info(f"SimilarityGenerator initialized successfully with model from {model_path}")

    def _load_model(self):
        """Loads the Word2Vec model from the specified path."""
        if not self.model_path.exists():
            self.logger.error(f"Word2Vec model file not found at: {self.model_path}")
            return None
        try:
            self.logger.info("Loading Word2Vec model... (This may take a moment)")
            model = Word2Vec.load(str(self.model_path))
            self.logger.info("Word2Vec model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load Word2Vec model: {e}", exc_info=True)
            return None

    def generate_distractors(self, target_word_surface, sentence_with_blank, num_distractors=5):
        """
        Generates semantically similar distractors for a given target word.

        Args:
            target_word_surface (str): The surface form of the target word.
            num_distractors (int): The number of distractors to return.

        Returns:
            list: A list of semantically similar distractor words, or an empty list.
        """
        self.logger.info(f"--- Generating similarity distractors for '{target_word_surface}' ---")
        if not self.model:
            self.logger.error("Model not loaded. Cannot generate distractors.")
            return []


        target_info = self.processor.get_target_info_in_context(sentence_with_blank, target_word_surface)
        if not target_info:
            self.logger.warning(f"Could not analyze target word '{target_word_surface}'.")
            return []
        
        # First we try the surface form, if the surface form was not used for training we try the baseform (lemma) as a fallback
        target_token = target_word_surface
        if target_token not in self.model.wv:
            self.logger.warning(f"Surface form '{target_token}' not in Word2Vec vocabulary. Trying lemma...")
            target_token = target_info['base_form']
            if target_token not in self.model.wv:
                self.logger.error(f"Neither surface form '{target_word_surface}' nor lemma '{target_token}' are in the Word2Vec vocabulary.")
                return []
        
        # Find the most similar words using the model.
        try:
            # The model returns a list of (word, similarity_score) tuples.
            similar_items = self.model.wv.most_similar(target_token, topn=num_distractors)
            
            distractors_with_scores_str = ", ".join([f"'{word}' ({score:.5f})" for word, score in similar_items])
            self.logger.info(f"Found similar items for '{target_token}': [{distractors_with_scores_str}]")

            # We only return the word not the similarity score
            distractors = [word for word, score in similar_items]
            self.logger.info(f"Successfully generated {len(distractors)} distractors for target token '{target_token}'.")
            return distractors
        except Exception as e:
            self.logger.error(f"An error occurred while finding similar words for '{target_token}': {e}")
            return []
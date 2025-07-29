import logging
import re
from collections import defaultdict


class BaselineGenerator:
    """
    Generates distractors based on the "Baseline" method described in the paper.
    
    The criteria are:
    1. Same Part-of-Speech (POS) as the target word.
    2. Similar difficulty level, approximated by word frequency in a corpus.
    
    Distractors are ranked by the proximity of their frequency to the target word's frequency.
    """

    def __init__(self, corpus_processor, all_words_details):
        """
        Initializes the BaselineGenerator.

        Args:
            corpus_processor (CorpusProcessor): An initialized CorpusProcessor instance,
                                                used to analyze target words.
            all_words_details (dict): The dictionary loaded from the processed corpus data,
                                      mapping (lemma, pos_major) string keys to their details.
        """
        self.logger = logging.getLogger('BaselineGenerator')
        self.processor = corpus_processor
        self.words_data = all_words_details

        # Pre-process the words data by organizing it by POS for efficient lookups.
        # This avoids iterating through the entire vocabulary for every generation request.
        self.pos_to_words_data_organized = self._organize_by_pos()
        self.logger.info(
            f"Generator initialized. Organized vocabulary into {len(self.pos_to_words_data_organized)} POS categories.")

    def _clean_lemma(self, lemma: str) -> str:
        """
        Cleans a lemma by removing a hyphen and suffix if the suffix is 80%+ Latin characters.
        Example: 'タイム-time' -> 'タイム'
        """
        if '-' in lemma:
            parts = lemma.split('-', 1)
            if len(parts) > 1:
                main_part, suffix = parts
                if suffix:
                    # Find all ASCII letters in the suffix
                    latin_chars = re.findall(r'[a-zA-Z]', suffix)
                    # Check if the ratio of Latin characters is 80% or more
                    if (len(latin_chars) / len(suffix)) >= 0.8:
                        return main_part
        return lemma

    def _organize_by_pos(self):
        """
        Helper method to restructure the flat words_data dictionary into a dictionary
        grouped by major POS tags.

        Returns:
            defaultdict(list): A dictionary where keys are POS tags (e.g., '名詞') and
                               values are lists of (lemma, frequency) tuples.
        """
        self.logger.info("Organizing corpus vocabulary by Part-of-Speech for faster lookups...")
        # We use defaultdict since it will create an empty list object if we try to access an key that is not present
        pos_to_words_map = defaultdict(list)

        # The keys in self.words_data are tuples like ('猫', '名詞')
        # We iterate through the items (key-value pairs) of the dictionary.
        for key_tuple, details in self.words_data.items():
            # Ensure the key is a tuple with exactly two elements.
            if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                # Unpack the tuple directly into lemma and pos_major.
                lemma, pos_major = key_tuple
                # Get the frequency from the details dictionary.
                frequency = details.get('frequency', 0)
                # Only consider words that actually appeared in the corpus.
                if frequency > 0:
                    # Append the (lemma, frequency) tuple to the list for the corresponding POS tag.
                    pos_to_words_map[pos_major].append((lemma, frequency))
            else:
                # Log a warning if a key is not in the expected format.
                self.logger.warning(f"Skipping malformed key during organization: '{key_tuple}'")

        self.logger.info("Finished organizing vocabulary.")

        return pos_to_words_map

    def generate_distractors(self, target_word_surface, sentence_with_blank, num_distractors=5):
        """
        Generates a ranked list of distractors for a given target word.

        Args:
            target_word_surface (str): The surface form of the word to generate distractors for (e.g., "走った").
            sentence_with_blank (str): The carrier sentence containing a blank token
            num_distractors (int): The number of distractors to return.

        Returns:
            list: A list of distractor words (lemmas), or an empty list if generation fails.
        """
        self.logger.info(f"--- Generating baseline distractors for '{target_word_surface}' ---")

        # 1. Analyze the target word to get its lemma and POS.
        # target_info = self.processor.get_token_info_for_word(target_word_surface)
        target_info = self.processor.get_target_info_in_context(sentence_with_blank, target_word_surface)

        if not target_info:
            self.logger.error(f"Could not analyze target word '{target_word_surface}'. Aborting.")
            return []

        target_lemma = target_info['base_form']
        target_pos = target_info['pos_major']

        target_key_tuple = (target_lemma, target_pos)

        # Look up the target's details using the correct TUPLE key.
        target_details = self.words_data.get(target_key_tuple)

        if not target_details:
            self.logger.error(
                f"Target word key ('{target_lemma}', '{target_pos}') not found in the processed corpus data.")
            return []

        target_frequency = target_details.get('frequency', 0)
        self.logger.info(f"Target analyzed: Lemma='{target_lemma}', POS='{target_pos}', Frequency={target_frequency}")

        # 2. Get all candidate words with the same POS.
        candidate_pool = self.pos_to_words_data_organized.get(target_pos)
        if not candidate_pool:
            self.logger.error(f"No words found in the corpus with POS '{target_pos}'. Cannot generate distractors.")
            return []

        self.logger.info(f"Found {len(candidate_pool)} candidate words with POS '{target_pos}'.")

        # 3. Calculate frequency difference for each candidate and filter out the target itself.
        candidates_with_diff = []
        for cand_lemma, cand_freq in candidate_pool:
            # Make sure we do not return the target word as a distractor
            if cand_lemma != target_lemma:
                freq_difference = abs(cand_freq - target_frequency)
                candidates_with_diff.append((cand_lemma, freq_difference))

        # 4. Sort candidates by the frequency difference (the smaller, the better).
        sorted_candidates = sorted(candidates_with_diff, key=lambda item: item[1])

        # 5. Extract, clean, and de-duplicate the top N distractors.
        distractors = []
        seen_lemmas = {self._clean_lemma(target_lemma)}
        for lemma, diff in sorted_candidates:
            cleaned_lemma = self._clean_lemma(lemma)
            if cleaned_lemma not in seen_lemmas:
                distractors.append(cleaned_lemma)
                seen_lemmas.add(cleaned_lemma)
            if len(distractors) >= num_distractors:
                break
        # distractors = [lemma for lemma, diff in sorted_candidates[:num_distractors]]

        self.logger.info(f"Successfully generated {len(distractors)} distractors: {distractors}")
        return distractors

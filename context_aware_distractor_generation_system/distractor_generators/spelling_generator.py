import logging
from collections import defaultdict

class SpellingGenerator:
    """
    Generates distractors based on the "Spelling Similarity" (+Spell) method.
    
    The criteria are:
    1. The candidate must share at least one character with the target word's lemma.
    2. The candidate must have the same Part-of-Speech (POS) as the target word.
    3. Candidates are ranked by the proximity of their word frequency to the target word's frequency.
    """
    
    def __init__(self, corpus_processor, all_words_details):
        """
        Initializes the SpellingGenerator.

        Args:
            corpus_processor (CorpusProcessor): An initialized CorpusProcessor instance.
            all_words_details (dict): The dictionary from the processed corpus data,
                                      mapping (lemma, pos_major) tuples to their details.
        """
        self.logger = logging.getLogger('SpellingGenerator')
        self.processor = corpus_processor
        self.words_data = all_words_details
        
        # Pre-build an inverted index for fast character-based lookups.
        self.char_to_words_index = self._create_char_to_words_index()
        self.logger.info(f"Generator initialized. Built character index for {len(self.char_to_words_index)} unique characters.")

    def _create_char_to_words_index(self):
        """
        Creates an inverted index mapping each character to a list of (lemma, pos_major)
        keys that contain that character.
        
        Returns:
            defaultdict(list): A dictionary where keys are characters and values are lists of word keys.
        """
        self.logger.info("Building character-to-word inverted index for fast lookups...")
        # Instead of looking up a word to see its characters, we want to look up a character and instantly get a list of all words that contain it. 
        index = defaultdict(list)
        for key_tuple, details in self.words_data.items():
            # key_tuple is e.g., ('猫', '名詞')
            lemma = key_tuple[0]
            # Create a set of unique characters in the lemma
            unique_chars_in_lemma = set(lemma)
            for char in unique_chars_in_lemma:
                index[char].append(key_tuple)
        
        self.logger.info("Finished building character index.")
        return index

    def generate_distractors(self, target_word_surface, sentence_with_blank, num_distractors=5):
        """
        Generates a ranked list of distractors based on spelling similarity.

        Args:
            target_word_surface (str): The surface form of the word.
            num_distractors (int): The number of distractors to return.

        Returns:
            list: A list of distractor words (lemmas), or an empty list.
        """
        self.logger.info(f"--- Generating spelling distractors for '{target_word_surface}' ---")

        # 1. Analyze the target word to get its lemma, POS, and frequency.
        # target_info = self.processor.get_token_info_for_word(target_word_surface)
        target_info = self.processor.get_target_info_in_context(sentence_with_blank, target_word_surface)
        if not target_info:
            self.logger.error(f"Could not analyze target word '{target_word_surface}'. Aborting.")
            return []
        
        target_lemma = target_info['base_form']
        target_pos = target_info['pos_major']
        target_key = (target_lemma, target_pos)
        
        target_details = self.words_data.get(target_key)
        if not target_details:
            self.logger.error(f"Target key '{target_key}' not found in corpus data.")
            return []
            
        target_frequency = target_details.get('frequency', 0)
        self.logger.info(f"Target analyzed: Lemma='{target_lemma}', POS='{target_pos}', Frequency={target_frequency}")

        # 2. Find all candidate words that share at least one character.
        target_chars = set(target_lemma)
        self.logger.info(f"Unique characters in target lemma: {target_chars}")
        
        candidate_keys = set()
        for char in target_chars:
            # Use the pre-built index for a very fast lookup
            if char in self.char_to_words_index:
                candidate_keys.update(self.char_to_words_index[char])
        
        self.logger.info(f"Found {len(candidate_keys)} potential candidates from character index.")

        # 3. Filter candidates by POS and calculate frequency difference.
        candidates_with_diff = []
        for cand_key in candidate_keys:
            cand_lemma, cand_pos = cand_key
            
            # Filter out the target word itself and words with a different POS.
            if cand_lemma == target_lemma or cand_pos != target_pos:
                continue
            
            cand_details = self.words_data.get(cand_key)
            if not cand_details:
                continue # Check if the candidate is in our corpus -> that should always be the case
                
            cand_freq = cand_details.get('frequency', 0)
            freq_difference = abs(cand_freq - target_frequency)
            candidates_with_diff.append((cand_lemma, freq_difference))
            
        # 4. Sort candidates by frequency difference (smallest difference is best).
        sorted_candidates = sorted(candidates_with_diff, key=lambda item: item[1])
        
        # 5. Extract the top N distractors.
        distractors = [lemma for lemma, diff in sorted_candidates[:num_distractors]]
        
        self.logger.info(f"Successfully generated {len(distractors)} distractors: {distractors}")
        return distractors
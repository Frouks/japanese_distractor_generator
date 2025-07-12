import logging
import math
from collections import defaultdict
import json
from pathlib import Path

class CooccurrenceGenerator:
    """
    Generates distractors based on the "+Co-occur" method from the paper.
    
    The criteria are:
    1. A distractor must often co-occur with the target word in the corpus.
    2. The strength of co-occurrence is measured by Pointwise Mutual Information (PMI).
    3. We add a constraint that the distractor should have the same Part-of-Speech (POS)
       as the target word to ensure syntactic plausibility.
    """
    
    def __init__(self, corpus_processor, all_words_details, cooccurrence_data, total_sentences):
        """
        Initializes the CooccurrenceGenerator.

        Args:
            corpus_processor (CorpusProcessor): An initialized CorpusProcessor instance.
            all_words_details (dict): Dictionary from processed corpus data,
                                      mapping (lemma, pos_major) tuples to details.
            cooccurrence_data (dict): A dictionary containing co-occurrence statistics.
                                      Expected keys: 'counts', 'total_pairs'.
            total_sentences (int): The total number of sentences in the corpus.
        """
        self.logger = logging.getLogger('CooccurrenceGenerator')
        self.processor = corpus_processor
        self.words_data = all_words_details
        
        # Unpack co-occurrence data
        self.cooccurrence_counts = cooccurrence_data.get('counts', {})
        self.total_cooccurrence_pairs = cooccurrence_data.get('total_pairs', 0)
        
        # Pre-calculate total word count for PMI
        self.total_word_count = sum(details.get('frequency', 0) for details in self.words_data.values())
        
        self.total_sentences = total_sentences

        if self.total_word_count == 0 or self.total_cooccurrence_pairs == 0:
            self.logger.error("Total word count or total co-occurrence pairs is zero. PMI cannot be calculated.")
            self.cooccurrence_counts = {} # Prevent further operations

        # For faster lookups, build an inverted index from a word to its co-occurring partners
        self.cooccurrence_index = self._build_cooccurrence_index()
        self.logger.info(f"Generator initialized with {len(self.cooccurrence_counts)} co-occurrence pairs and an index of {len(self.cooccurrence_index)} lemmas.")

    def _build_cooccurrence_index(self):
        """Builds an inverted index from a word to its co-occurring pairs and counts."""
        self.logger.info("Building co-occurrence inverted index for fast lookups...")
        index = defaultdict(list)
        for (w1, w2), count in self.cooccurrence_counts.items():
            index[w1].append((w2, count))
            index[w2].append((w1, count))
        self.logger.info("Finished building co-occurrence index.")
        return index

    @staticmethod
    def load_cooccurrence_data(file_path_str):
        """Loads co-occurrence data from a JSON file."""
        logger = logging.getLogger('CooccurrenceGenerator.Loader')
        file_path = Path(file_path_str)
        if not file_path.exists():
            logger.error(f"Co-occurrence data file not found: {file_path}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string keys "word1|word2" back to tuples ('word1', 'word2')
            reformatted_counts = {tuple(key.split('|')): count for key, count in data['counts'].items()}
            data['counts'] = reformatted_counts

            logger.info(f"Successfully loaded {len(data['counts'])} co-occurrence counts from {file_path}")
            return data
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Failed to load or parse co-occurrence data from {file_path}: {e}", exc_info=True)
            return None

    def _calculate_pmi(self, p_x, p_y, p_xy):
        """Calculates PMI, handling potential zero probabilities."""
        if p_x <= 0 or p_y <= 0 or p_xy <= 0:
            return -float('inf')  # PMI is undefined, effectively lowest possible score
        
        try:
            # log2(p(x,y) / (p(x) * p(y)))
            return math.log2(p_xy / (p_x * p_y))
        except (ValueError, ZeroDivisionError):
            return -float('inf')

    def generate_distractors(self, target_word_surface, sentence_with_blank, num_distractors=5):
        """
        Generates a ranked list of distractors based on PMI using sentence-level probabilities.
        """
        self.logger.info(f"--- Generating co-occurrence distractors for '{target_word_surface}' ---")
        if not self.cooccurrence_index:
            self.logger.error("No co-occurrence index available. Aborting.")
            return []

        # 1. Analyze the target word
        #target_info = self.processor.get_token_info_for_word(target_word_surface)
        target_info = self.processor.get_target_info_in_context(sentence_with_blank, target_word_surface)
        if not target_info:
            self.logger.error(f"Could not analyze target word '{target_word_surface}'.")
            return []
        
        target_lemma = target_info['base_form']
        target_pos = target_info['pos_major']
        target_key = (target_lemma, target_pos)

        target_details = self.words_data.get(target_key)
        if not target_details:
            self.logger.error(f"Target key '{target_key}' not found in corpus data.")
            return []

        # We will use the total token frequency as a proxy for sentence frequency.
        target_freq = target_details.get('frequency', 0)
        # P(target) ≈ Count(target) / Total Sentences
        p_target = target_freq / self.total_sentences if self.total_sentences > 0 else 0
        self.logger.info(f"Target: Lemma='{target_lemma}', POS='{target_pos}', Freq={target_freq}")

        # 2. Find co-occurring words and calculate PMI
        co_occurring_surfaces = self.cooccurrence_index.get(target_word_surface, [])
        if not co_occurring_surfaces:
            co_occurring_surfaces = self.cooccurrence_index.get(target_lemma, [])
        if not co_occurring_surfaces:
            self.logger.warning(f"Target '{target_word_surface}' not found in co-occurrence index.")
            return []
        
        self.logger.info(f"Found {len(co_occurring_surfaces)} co-occurring candidate surfaces to analyze.")

        candidates_with_pmi = []
        processed_lemmas = set() 

        for cand_surface, co_occur_freq in co_occurring_surfaces:
            cand_info = self.processor.get_token_info_for_word(cand_surface)
            if not cand_info:
                continue

            cand_lemma = cand_info['base_form']
            cand_pos = cand_info['pos_major']
            
            if cand_pos != target_pos or cand_lemma in processed_lemmas or cand_lemma == target_lemma:
                continue
            
            processed_lemmas.add(cand_lemma)
            cand_key = (cand_lemma, cand_pos)
            cand_details = self.words_data.get(cand_key)

            if not cand_details:
                continue
            
            # P(candidate) ≈ Count(candidate) / Total Sentences
            cand_freq = cand_details.get('frequency', 0)
            p_cand = cand_freq / self.total_sentences if self.total_sentences > 0 else 0

            # P(target, candidate) = Count(target, candidate) / Total Sentences
            p_target_cand = co_occur_freq / self.total_sentences if self.total_sentences > 0 else 0
        
            pmi_score = self._calculate_pmi(p_target, p_cand, p_target_cand)
        
            if pmi_score > -float('inf'):
                candidates_with_pmi.append((cand_lemma, pmi_score))

        # 4. Sort and return
        sorted_candidates = sorted(candidates_with_pmi, key=lambda item: item[1], reverse=True)
        distractors = [lemma for lemma, score in sorted_candidates[:num_distractors]]
        
        self.logger.info(f"Successfully generated {len(distractors)} distractors: {distractors}")
        return distractors
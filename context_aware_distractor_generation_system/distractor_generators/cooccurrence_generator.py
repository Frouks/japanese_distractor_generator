import json
import logging
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Union

# This makes the script runnable from anywhere by locating its parent project directories
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for environments where __file__ might not be defined
    project_root = Path('.').resolve().parents[2]

# Define the data directory relative to the located project root
PROCESSED_DATA_DIR = project_root / "re_implementation/processed_corpus_data"
_COOCCURRENCE_INDEX_FILE = PROCESSED_DATA_DIR / "jp_cooccurrence_index.pkl"

from context_aware_distractor_generation_system.constants.SentenceContextEnum import SentenceContextEnum


class CooccurrenceGenerator:
    """
    Generates distractors based on co-occurrence, adapting its method based on context.
    - Closed Context: Uses standard PMI to find domain-specific pairs.
    - Open Context: Uses PMI^k (specifically PMI^2) to find more general pairs.
    """

    def __init__(self, corpus_processor, all_words_details, cooccurrence_data, total_sentences):
        """
        Initializes the CooccurrenceGenerator.
        """
        self.logger = logging.getLogger('CooccurrenceGenerator')
        self.processor = corpus_processor
        self.words_data = all_words_details
        self.cooccurrence_counts = cooccurrence_data.get('counts', {})
        self.total_cooccurrence_pairs = cooccurrence_data.get('total_pairs', 0)
        self.total_sentences = total_sentences

        if self.total_cooccurrence_pairs == 0:
            self.logger.error("Total co-occurrence pairs is zero. PMI cannot be calculated.")
            self.cooccurrence_counts = {}

        if _COOCCURRENCE_INDEX_FILE.exists():
            self.logger.info(f"Loading cached co-occurrence index from {_COOCCURRENCE_INDEX_FILE}...")
            with open(_COOCCURRENCE_INDEX_FILE, 'rb') as f_in:
                self.cooccurrence_index = pickle.load(f_in)
            self.logger.info("✅ Cached index loaded successfully.")
        else:
            self.logger.warning("Cached co-occurrence index not found. Building it now (this is a one-time process)...")
            self.logger.warning(f"Given Location: {_COOCCURRENCE_INDEX_FILE}")
            # Build the index from scratch
            self.cooccurrence_index = self._build_cooccurrence_index()
            # Save it for next time
            self._save_cooccurrence_index(self.cooccurrence_index)

        self.logger.info(
            f"Generator initialized with {len(self.cooccurrence_counts)} co-occurrence pairs and an index of {len(self.cooccurrence_index)} lemmas.")

    def _save_cooccurrence_index(self, index_obj):
        """Saves the generated index to a pickle file for future use."""
        self.logger.info(f"Saving co-occurrence index to {_COOCCURRENCE_INDEX_FILE} for future runs...")
        try:
            _COOCCURRENCE_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_COOCCURRENCE_INDEX_FILE, 'wb') as f_out:
                pickle.dump(index_obj, f_out)
            self.logger.info("✅ Index saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save co-occurrence index: {e}", exc_info=True)

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
        except Exception as e:
            logger.error(f"Failed to load or parse co-occurrence data from {file_path}: {e}", exc_info=True)
            return None

    def _calculate_pmi(self, p_x, p_y, p_xy):
        """
        Calculates standard Pointwise Mutual Information (PMI).
        PMI(x, y) = log2(p(x,y) / (p(x) * p(y)))
        """
        if p_x <= 0 or p_y <= 0 or p_xy <= 0:
            return -float('inf')
        #
        return math.log2(p_xy / (p_x * p_y))

    def _calculate_pmi_k(self, p_x, p_y, p_xy, k=2):
        """
        Calculates PMI^k to reduce bias towards low-frequency words.
        PMI^k(x, y) = PMI(x, y) − (− (k − 1) log(p(x, y))).
        """
        if p_x <= 0 or p_y <= 0 or p_xy <= 0:
            return -float('inf')

        # Standard PMI
        pmi_val = math.log2(p_xy / (p_x * p_y))
        correction_factor = (k - 1) * math.log2(p_xy)

        return pmi_val + correction_factor

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
                    latin_chars = re.findall(r'[a-zA-Z]', suffix)
                    if (len(latin_chars) / len(suffix)) >= 0.8:
                        return main_part
        return lemma

    def generate_distractors(self, target_word_surface, sentence_with_blank, context_type: SentenceContextEnum,
                             num_distractors=5, include_pmi_score: bool = True) -> Union[
        list[tuple[str, float]], list[str]]:
        """
        Generates a ranked list of distractors, choosing the PMI variant based on context.

        Args:
            target_word_surface (str): The surface form of the word.
            sentence_with_blank (str): The carrier sentence.
            context_type (SentenceContextEnum): The context (OPEN or CLOSED).
            num_distractors (int): The number of distractors to return.
            include_pmi_score (bool): If False (default), returns a list of words.
                                      If True, returns a list of (word, pmi_score) tuples.

        Returns:
            A list of distractors in the specified format.
        """
        context = "Open" if context_type == SentenceContextEnum.OPEN else "Closed"
        self.logger.info(
            f"--- Generating co-occurrence distractors for '{target_word_surface}' (Context: {context}) ---")
        if not self.cooccurrence_index:
            self.logger.error("No co-occurrence index available. Aborting.")
            return []

        target_info = self.processor.get_target_info_in_context(sentence_with_blank, target_word_surface)
        if not target_info:
            self.logger.error(f"Could not analyze target word '{target_word_surface}'.")
            return []

        target_lemma = target_info['base_form']
        target_pos = target_info['pos_major']
        target_key = (target_lemma, target_pos)
        target_details = self.words_data.get(target_key)
        if not target_details:
            return []

        target_freq = target_details.get('frequency', 0)
        p_target = target_freq / self.total_sentences if self.total_sentences > 0 else 0
        self.logger.info(f"Target: Lemma='{target_lemma}', POS='{target_pos}', Freq={target_freq}")

        co_occurring_surfaces = self.cooccurrence_index.get(target_word_surface, []) or self.cooccurrence_index.get(
            target_lemma, [])
        if not co_occurring_surfaces:
            self.logger.warning(f"Target '{target_word_surface}' not found in co-occurrence index.")
            return []

        self.logger.info(f"Found {len(co_occurring_surfaces)} co-occurring candidate surfaces to analyze.")

        candidates_with_scores = []
        processed_lemmas = {target_lemma}

        for cand_surface, co_occur_freq in co_occurring_surfaces:
            cand_info = self.processor.get_token_info_for_word(cand_surface)
            if not cand_info: continue

            cand_lemma = cand_info['base_form']
            cand_pos = cand_info['pos_major']

            if cand_pos != target_pos or cand_lemma in processed_lemmas: continue

            processed_lemmas.add(cand_lemma)
            cand_details = self.words_data.get((cand_lemma, cand_pos))
            if not cand_details: continue

            cand_freq = cand_details.get('frequency', 0)
            p_cand = cand_freq / self.total_sentences if self.total_sentences > 0 else 0
            p_target_cand = co_occur_freq / self.total_sentences if self.total_sentences > 0 else 0

            score = 0
            if context_type == SentenceContextEnum.CLOSED:
                score = self._calculate_pmi(p_target, p_cand, p_target_cand)
            elif context_type == SentenceContextEnum.OPEN:
                # Using PMI^2 (k=2) for open contexts -> results in more general distractors
                score = self._calculate_pmi_k(p_target, p_cand, p_target_cand, k=2)
            else:
                self.logger.warning(f"Invalid context type '{context_type}'. Defaulting to standard PMI.")
                score = self._calculate_pmi(p_target, p_cand, p_target_cand)

            if score > -float('inf'):
                candidates_with_scores.append((cand_lemma, score))

        sorted_candidates = sorted(candidates_with_scores, key=lambda item: item[1], reverse=True)
        # Clean and format the final output
        final_candidates_with_scores = []
        seen_lemmas = {self._clean_lemma(target_lemma)}
        for lemma, score in sorted_candidates:
            cleaned_lemma = self._clean_lemma(lemma)
            if cleaned_lemma not in seen_lemmas:
                final_candidates_with_scores.append((cleaned_lemma, score))
                seen_lemmas.add(cleaned_lemma)
            if len(final_candidates_with_scores) >= num_distractors:
                break

        if include_pmi_score:
            distractors = final_candidates_with_scores
        else:
            distractors = [lemma for lemma, score in final_candidates_with_scores]

        self.logger.info(f"Successfully generated {len(distractors)} distractors: {distractors}")
        return distractors

        self.logger.info(f"Successfully generated {len(distractors)} distractors: {distractors}")
        return distractors

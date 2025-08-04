import logging
import pickle
import time
from pathlib import Path

import spacy


class DistractorFilter:
    """
    Filters distractor candidates based on corpus-wide n-gram and dependency data.
    """

    def __init__(self, trigram_path, dependency_index_path):
        """
        Initializes the filter by loading pre-built data sets.

        Args:
            trigram_path (str or Path): Path to the pickled trigram set.
            dependency_index_path (str or Path): Path to the pickled dependency INVERTED INDEX.
        """
        self.logger = logging.getLogger('DistractorFilter')

        self.trigrams = self._load_pickle_data(
            Path(trigram_path),
            "trigrams",
            empty_type=set()
        )

        self.dependency_index = self._load_pickle_data(
            Path(dependency_index_path),
            "dependency index",
            empty_type={}
        )

        self.logger.info("Loading spaCy/GiNZA model for live parsing...")
        try:
            self.nlp = spacy.load("ja_ginza")
            self.logger.info("✅ GiNZA model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load GiNZA model: {e}", exc_info=True)
            self.nlp = None

    def _load_pickle_data(self, file_path, data_name, empty_type):
        """
        Helper method to load a pickled data object from a file.
        Returns a given empty object on failure.

        Args:
            file_path (Path): Path to the pickle file.
            data_name (str): Name for logging.
            empty_type: The empty object to return on failure (e.g., set(), {}).
        """
        if not file_path.exists():
            self.logger.error(f"{data_name.capitalize()} data file not found: {file_path}")
            return empty_type

        self.logger.info(f"Loading {data_name} from {file_path}...")
        start_time = time.time()
        try:
            with open(file_path, 'rb') as f:
                data_obj = pickle.load(f)
            duration = time.time() - start_time
            self.logger.info(
                f"Successfully loaded {len(data_obj):,} items into a {type(data_obj).__name__} in {duration:.2f}s.")
            return data_obj
        except Exception as e:
            self.logger.error(f"Failed to load {data_name} data: {e}", exc_info=True)
            return empty_type

    def filter_by_trigram(self, candidates, prev_word, next_word) -> tuple[list[str], list[str]]:
        """
        Applies the trigram filter.
        """
        if not self.trigrams:
            self.logger.warning("Trigram set is empty. Skipping trigram filtering.")
            return candidates

        accepted, rejected = [], []
        for candidate in candidates:
            trigram_to_check = (prev_word, candidate, next_word)
            if trigram_to_check not in self.trigrams:
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        rejected_count = len(candidates) - len(accepted)
        if rejected_count > 0:
            self.logger.info(f"Trigram filter rejected {rejected_count} candidates.")
        return accepted, rejected

    def filter_by_dependency(self, candidates, sentence_template, blank_placeholder="___") -> tuple[
        list[str], list[str]]:
        """
        Applies the dependency filter with morphology-aware lemma handling.
        Handles GiNZA's morphological splitting of inflected verbs and adjectives.
        """
        if not self.dependency_index or not self.nlp:
            self.logger.warning("Dependency index or GiNZA model not available. Skipping dependency filtering.")
            return candidates, []

        accepted, rejected = [], []

        for candidate in candidates:
            full_sentence = sentence_template.replace(blank_placeholder, candidate, 1)
            is_rejected = False

            try:
                doc = self.nlp(full_sentence)

                # Strategy 1: Look for exact matches (works for base forms like '猫', '寒い')
                candidate_lemmas = set()
                for token in doc:
                    if token.text == candidate:
                        candidate_lemmas.add(token.lemma_)

                # Strategy 2: Handle morphologically split candidates
                # Find content word stems that could be part of our inflected candidate
                if not candidate_lemmas:
                    for token in doc:
                        # Check if this could be the main stem of our candidate
                        # Examples from test results:
                        # - candidate="書いた", token.text="書い", token.lemma_="書く" (VERB)
                        # - candidate="寒かった", token.text="寒かっ", token.lemma_="寒い" (ADJ)
                        # - candidate="面白くない", token.text="面白く", token.lemma_="面白い" (ADJ)
                        if (token.pos_ in ['VERB', 'ADJ'] and
                                candidate.startswith(token.text)):
                            candidate_lemmas.add(token.lemma_)

                # Strategy 3: Fallback -> prevents system from crashing
                if not candidate_lemmas:
                    candidate_lemmas.add(candidate)

                # Check dependency relations involving the candidate
                for token in doc:
                    # Only process tokens where candidate is involved
                    if (token.lemma_ in candidate_lemmas or
                            token.head.lemma_ in candidate_lemmas):

                        # Create the dependency relation
                        live_relation = (token.dep_, token.head.lemma_, token.lemma_)

                        # Check if this relation exists in our corpus
                        corpus_relations = self.dependency_index.get(token.lemma_, [])

                        if live_relation in corpus_relations:
                            is_rejected = True
                            break

                if not is_rejected:
                    accepted.append(candidate)
                else:
                    rejected.append(candidate)

            except Exception as e:
                self.logger.error(f"Error during dependency parsing for candidate '{candidate}': {e}", exc_info=True)
                accepted.append(candidate)

        rejected_count = len(rejected)
        if rejected_count > 0:
            self.logger.info(f"Dependency filter rejected {rejected_count} candidates.")
        return accepted, rejected

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
        Applies the dependency filter.
        """
        if not self.dependency_index or not self.nlp:
            self.logger.warning("Dependency index or GiNZA model not available. Skipping dependency filtering.")
            return candidates

        accepted, rejected = [], []
        for candidate in candidates:
            full_sentence = sentence_template.replace(blank_placeholder, candidate, 1)
            is_rejected = False
            try:
                doc = self.nlp(full_sentence)
                corpus_relations = set()
                for token in doc:
                    known_rels_for_token = self.dependency_index.get(token.lemma_, [])
                    if known_rels_for_token:
                        corpus_relations.update(known_rels_for_token)

                if not corpus_relations:
                    accepted.append(candidate)
                    continue

                for token in doc:
                    if token.lemma_ != candidate and token.head.lemma_ != candidate:
                        continue

                    live_relation = (token.dep_, token.head.lemma_, token.lemma_)

                    if live_relation in corpus_relations:
                        is_rejected = True
                        break

                if not is_rejected:
                    accepted.append(candidate)
                else:
                    rejected.append(candidate)  # CHANGED: Populate the rejected list

            except Exception as e:
                self.logger.error(f"Error during dependency parsing for candidate '{candidate}': {e}", exc_info=True)
                accepted.append(candidate)

        rejected_count = len(candidates) - len(accepted)
        if rejected_count > 0:
            self.logger.info(f"Dependency filter rejected {rejected_count} candidates.")
        return accepted, rejected

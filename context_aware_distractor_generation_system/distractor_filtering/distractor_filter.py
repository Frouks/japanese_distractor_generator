import logging
import pickle
from pathlib import Path
import time
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
            self.logger.info(f"Successfully loaded {len(data_obj):,} items into a {type(data_obj).__name__} in {duration:.2f}s.")
            return data_obj
        except Exception as e:
            self.logger.error(f"Failed to load {data_name} data: {e}", exc_info=True)
            return empty_type

    def filter_by_trigram(self, candidates, prev_word, next_word):
        """
        Applies the trigram filter.
        """
        if not self.trigrams:
            self.logger.warning("Trigram set is empty. Skipping trigram filtering.")
            return candidates

        reliable_candidates = []
        for candidate in candidates:
            trigram_to_check = (prev_word, candidate, next_word)
            if trigram_to_check not in self.trigrams:
                reliable_candidates.append(candidate)
        
        rejected_count = len(candidates) - len(reliable_candidates)
        if rejected_count > 0:
            self.logger.info(f"Trigram filter rejected {rejected_count} candidates.")
        return reliable_candidates
        
    def filter_by_dependency(self, candidates, sentence_template, blank_placeholder="___"):
        """
        Applies the dependency filter.
        """
        if not self.dependency_index or not self.nlp:
            self.logger.warning("Dependency index or GiNZA model not available. Skipping dependency filtering.")
            return candidates

        reliable_candidates = []
        for candidate in candidates:
            # Form the full sentence with the candidate
            full_sentence = sentence_template.replace(blank_placeholder, candidate, 1)
            
            is_rejected = False
            try:
                # Parse the sentence
                doc = self.nlp(full_sentence)
                
                # --- MODIFIED: More efficient and robust checking logic ---
                
                # 1. Get all relations that exist in our pre-built index for the words in this sentence.
                # This is a fast preliminary check.
                corpus_relations = set()
                for token in doc:
                    # For each word in the sentence, get its list of known common relations from our index.
                    known_rels_for_token = self.dependency_index.get(token.lemma_, [])
                    if known_rels_for_token:
                        # Add them to a temporary set for fast lookups later.
                        corpus_relations.update(known_rels_for_token)
                
                # If no word in the sentence has any known common relations, we can't filter.
                if not corpus_relations:
                    reliable_candidates.append(candidate)
                    continue

                # 2. Now, generate the relations for the *current* sentence and check against our known set.
                for token in doc:
                    # We only care about relations involving our candidate word.
                    if token.lemma_ != candidate and token.head.lemma_ != candidate:
                        continue
                    
                    # Form the relation tuple from the live parse.
                    live_relation = (token.dep_, token.head.lemma_, token.lemma_)
                    
                    # Check if this freshly parsed relation exists in our set of known common relations.
                    if live_relation in corpus_relations:
                        self.logger.debug(f"Rejecting '{candidate}' due to attested relation: {live_relation}")
                        is_rejected = True
                        break # Found a match, no need to check further for this candidate.
                
                if not is_rejected:
                    reliable_candidates.append(candidate)

            except Exception as e:
                self.logger.error(f"Error during dependency parsing for candidate '{candidate}': {e}", exc_info=True)
                reliable_candidates.append(candidate)
        
        rejected_count = len(candidates) - len(reliable_candidates)
        if rejected_count > 0:
            self.logger.info(f"Dependency filter rejected {rejected_count} candidates.")
        return reliable_candidates
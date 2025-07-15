import json
from collections import defaultdict
import MeCab
import unidic_lite 

class CorpusProcessor:
    """
    Loads and processes corpus data and provides an interface to the MeCab tokenizer.
    This version explicitly initializes MeCab to be self-contained and portable.
    """
    def __init__(self, all_words_details_path: str):
        print("Initializing CorpusProcessor...")
        self.tagger = self._initialize_mecab()
        self.all_words_details = self._load_word_details(all_words_details_path)
        self.pos_vocab = self._organize_vocab_by_pos()

    def _initialize_mecab(self):
        """Initializes the MeCab Tagger with the self-contained unidic-lite dictionary."""
        try:
            mecab_args = f"-d {unidic_lite.DICDIR}"
            print(f"MeCab arguments: {mecab_args}")
            tagger = MeCab.Tagger(mecab_args)
            print("✅ MeCab Tagger initialized successfully using unidic-lite dictionary.")
            return tagger
        except RuntimeError as e:
            print(f"❌ CRITICAL: Failed to initialize MeCab Tagger. Error: {e}")
            return None

    def _load_word_details(self, file_path: str) -> dict:
        """Loads the JSON file containing all word details."""
        print(f"Loading word details from '{file_path}'...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # The keys in the JSON are strings, e.g., "猫|名詞"
                # We need to convert them back to tuples, e.g., ("猫", "名詞")
                string_keyed_data = json.load(f)
                tuple_keyed_data = {
                    tuple(key.split('|', 1)): value
                    for key, value in string_keyed_data.items()
                }
                print(f"✅ Successfully loaded {len(tuple_keyed_data)} word entries.")
                return tuple_keyed_data
        except FileNotFoundError:
            print(f"❌ CORPUS DATA NOT FOUND at '{file_path}'.")
            return {}

    def _organize_vocab_by_pos(self) -> defaultdict:
        """Groups the vocabulary by Part-of-Speech for fast lookups."""
        pos_vocab = defaultdict(list)
        if not self.all_words_details:
            return pos_vocab
            
        for (lemma, pos), details in self.all_words_details.items():
            pos_vocab[pos].append(lemma)
        return pos_vocab
        
    def get_word_details(self, word_lemma: str, pos: str) -> dict | None:
        """Safely gets details for a specific word."""
        return self.all_words_details.get((word_lemma, pos))

    def get_token_info(self, word_surface: str) -> dict | None:
        """Parses a single word to get its lemma and POS."""
        if not self.tagger: return None
        node = self.tagger.parseToNode(word_surface)
        node = node.next # Skip BOS
        if node and node.surface:
            features = node.feature.split(',')
            # Using UniDic feature indices
            pos = features[0]
            lemma = features[7] if len(features) > 7 else word_surface
            return {'lemma': lemma, 'pos': pos}
        return None
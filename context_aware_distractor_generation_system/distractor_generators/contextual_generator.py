import logging
import sys
from pathlib import Path
from typing import Union

import torch
from transformers import BertTokenizer, BertForMaskedLM

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from context_aware_distractor_generation_system.constants.SentenceContextEnum import SentenceContextEnum


class ContextualGenerator:
    """
    Generates distractors using a pre-trained Japanese BERT model to find
    contextually relevant candidates for a masked word in a sentence.
    """

    def __init__(self, model_name: str = 'cl-tohoku/bert-base-japanese-whole-word-masking'):
        """
        Initializes the generator by loading the BERT model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained BERT model
        """
        print("Initializing ContextualCandidateGenerator...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.model.eval()
            self.logger = logging.getLogger('ContextualGenerator')
            print(f"✅ BERT model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load BERT model '{model_name}'. Error: {e}")
            self.tokenizer = None
            self.model = None

    def generate_distractors(
            self,
            masked_sentence: str,
            target_word: str,
            context_type: SentenceContextEnum,
            top_n: int = 5,
            include_prob_score: bool = True
    ) -> Union[list[tuple[str, float]], list[str]]:
        """
        Generates distractors by predicting the masked token in a sentence.

        Args:
            masked_sentence (str): The carrier sentence with the target word replaced by '[MASK]'.
            target_word (str): The original target word (to be excluded from results).
            context_type (SentenceContextEnum): The type of context, 'open' or 'closed'.
            top_n (int): The final number of distractors to return.
            include_prob_score (bool): If False, returns only a list of words.
                                       If True, returns a list of (word, score) tuples.

        Returns:
            A list of distractors in the specified format.
        """
        if not self.model:
            print("Error: Model not loaded. Cannot generate contextual distractors.")
            return []

        self.logger.info(f"--- Generating contextual distractors for '{target_word}' (Context: {context_type}) ---")

        if context_type == SentenceContextEnum.OPEN:
            # For open contexts, accept a wider range of less probable but plausible words.
            min_prob, max_prob = 0.005, 0.1
        elif context_type == SentenceContextEnum.CLOSED:
            # For closed contexts, demand higher probability candidates.
            min_prob, max_prob = 0.1, 0.9
        else:
            print(f"Warning: Invalid context_type '{context_type}'. Defaulting to a wide range.")
            min_prob, max_prob = 0.005, 0.95

        print(
            f"\n--- Generating for mask in '{masked_sentence}' (Context: {context_type}, Prob Range: {min_prob}-{max_prob}) ---")

        # 2. Tokenize the input sentence
        tokenized_input = self.tokenizer(masked_sentence, return_tensors="pt")
        mask_token_index = torch.where(tokenized_input.input_ids == self.tokenizer.mask_token_id)[1]

        if len(mask_token_index) == 0:
            print("Warning: [MASK] token not found in the sentence.")
            return []

        # 3. Get model predictions
        with torch.no_grad():
            outputs = self.model(**tokenized_input)
            predictions = outputs.logits

        # 4. Get probabilities for the masked token
        mask_predictions = predictions[0, mask_token_index.item()]
        softmax_probs = torch.nn.functional.softmax(mask_predictions, dim=0)
        top_k_probs, top_k_indices = torch.topk(softmax_probs, k=200, dim=0)

        # 5. Filter and collect distractors
        distractors = []
        for prob, token_id in zip(top_k_probs, top_k_indices):
            prob = prob.item()
            if min_prob <= prob <= max_prob:
                token = self.tokenizer.decode(token_id)
                # Exclude the original target, the MASK token, and any sub-word tokens
                if token != target_word and not token.startswith('##') and token != '[UNK]':
                    distractors.append((token, prob))

        final_distractors_with_scores = distractors[:top_n]
        print(f"Successfully generated {len(final_distractors_with_scores)} distractors.")

        # Add the conditional return logic
        if include_prob_score:
            return final_distractors_with_scores
        else:
            return [word for word, score in final_distractors_with_scores]


if __name__ == '__main__':
    context_gen = ContextualGenerator()
    print("=" * 60)

    if context_gen.model:
        test_cases: list[dict[str, any]] = [
            {
                "sentence": "動物園で、大きな[MASK]が鼻を高く上げていた。",
                "target": "象",
                "context": SentenceContextEnum.CLOSED,
                "english": "At the zoo, the big ___ was raising its trunk high."
            },
            {
                "sentence": "公園で、たくさんの[MASK]が遊んでいた。",
                "target": "子供",
                "context": SentenceContextEnum.OPEN,
                "english": "At the park, many ___ were playing."
            },
            {
                "sentence": "彼は100メートルを10秒で[MASK]ことができる。",
                "target": "走る",
                "context": SentenceContextEnum.CLOSED,
                "english": "He can ___ 100 meters in 10 seconds."
            },
            {
                "sentence": "この[MASK]はとても重要です。",
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

            distractors = context_gen.generate_distractors(
                masked_sentence=case['sentence'],
                target_word=case['target'],
                context_type=case['context']
            )

            print("\n   Generated Distractors:")
            if distractors:
                for word, score in distractors:
                    print(f"     - {word:<15} (Probability: {score:.4f})")
            else:
                print("     No distractors found in the specified probability range.")
            print("=" * 60)

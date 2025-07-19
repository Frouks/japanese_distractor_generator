import torch
from transformers import BertTokenizer, BertForMaskedLM


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
            print(f"✅ BERT model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load BERT model '{model_name}'. Error: {e}")
            self.tokenizer = None
            self.model = None

    def generate_distractors(
            self,
            masked_sentence: str,
            target_word: str,
            context_type: str,
            top_n: int = 5
    ) -> list[tuple[str, float]]:
        """
        Generates distractors by predicting the masked token in a sentence.

        Args:
            masked_sentence (str): The carrier sentence with the target word replaced by '[MASK]'.
            target_word (str): The original target word (to be excluded from results).
            context_type (str): The type of context, 'open' or 'closed'.
            top_n (int): The final number of distractors to return.

        Returns:
            list[tuple[str, float]]: A list of (distractor, probability_score) tuples.
        """
        if not self.model:
            print("Error: Model not loaded. Cannot generate contextual distractors.")
            return []

        if context_type == 'Open':
            # For open contexts, accept a wider range of less probable but plausible words.
            min_prob, max_prob = 0.005, 0.1
        elif context_type == 'Closed':
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

        final_distractors = distractors[:top_n]
        print(f"Successfully generated {len(final_distractors)} distractors.")
        return final_distractors


if __name__ == '__main__':
    context_gen = ContextualGenerator()
    print("=" * 60)

    if context_gen.model:
        test_cases: list[dict[str, any]] = [
            {
                "sentence": "動物園で、大きな[MASK]が鼻を高く上げていた。",
                "target": "象",
                "context": "Closed",
                "english": "At the zoo, the big ___ was raising its trunk high."
            },
            {
                "sentence": "公園で、たくさんの[MASK]が遊んでいた。",
                "target": "子供",
                "context": "Open",
                "english": "At the park, many ___ were playing."
            },
            {
                "sentence": "彼は100メートルを10秒で[MASK]ことができる。",
                "target": "走る",
                "context": "Closed",
                "english": "He can ___ 100 meters in 10 seconds."
            },
            {
                "sentence": "この[MASK]はとても重要です。",
                "target": "問題",
                "context": "Open",
                "english": "This ___ is very important."
            }
        ]

        for i, case in enumerate(test_cases):
            print(f"🧪 TEST CASE {i + 1}: {case['context'].upper()} CONTEXT")
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

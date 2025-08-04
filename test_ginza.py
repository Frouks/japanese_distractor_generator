def test_ginza_tokenization():
    import spacy
    nlp = spacy.load("ja_ginza")

    # Test the exact cases from your log
    test_cases = [
        ("彼女は昨日、長い手紙を書いた。", "書いた"),
        ("彼はマラソン大会で速く走った。", "走った"),
        ("私の猫はとても可愛い。", "猫"),
        ("妻が買ったかばんを友達に⾒せたい。", "買った"),
        ("今日はとても寒かった。", "寒かった"),  # was cold
        ("彼女はとても美しかった。", "美しかった"),  # was beautiful
        ("この本は面白くない。", "面白くない"),  # not interesting
        ("昨日は暑くなかった。", "暑くなかった"),  # was not hot
        ("今日は寒い。", "寒い"),  # cold (basic form)
        ("日本の首都は東京です。", "東京")  # the capital of japan is tokyo
    ]

    for sentence, candidate in test_cases:
        print(f"\n=== Testing: '{candidate}' in '{sentence}' ===")
        doc = nlp(sentence)

        print("All tokens:")
        for token in doc:
            print(f"  '{token.text}' → lemma: '{token.lemma_}'")
            print(
                f" '{token.text}' -> dependencies: '{token.dep_}' with head: '{token.head.lemma_}' and child: '{token.lemma_}'")

        print(f"\nLooking for candidate '{candidate}':")
        found_exact = False
        for token in doc:
            if token.text == candidate:
                print(f"  ✅ EXACT MATCH: '{token.text}' → lemma: '{token.lemma_}'")
                found_exact = True

        if not found_exact:
            print(f"  ❌ NO EXACT MATCH for '{candidate}'")
            print("  → Need to use fallback strategy")


def test_morphological_complexity():
    import spacy
    nlp = spacy.load("ja_ginza")

    print("=== THREE MORPHOLOGICAL COMPLEXITY EXAMPLES ===\n")

    # Example 1: Already in lemma form (no splitting)
    print("🔹 EXAMPLE 1: Already in lemma form")
    sentence1 = "私は本を読む。"
    candidate1 = "本"
    print(f"Sentence: {sentence1}")
    print(f"Candidate: {candidate1}")

    doc1 = nlp(sentence1)
    print("Tokenization:")
    for token in doc1:
        print(f"  '{token.text}' → lemma: '{token.lemma_}' (POS: {token.pos_})")

    # Check if exact match works
    for token in doc1:
        if token.text == candidate1:
            print(f"✅ EXACT MATCH: '{token.text}' → lemma: '{token.lemma_}'")
    print()

    # Example 2: Split into 2 parts (stem + auxiliary)
    print("🔹 EXAMPLE 2: Split into 2 parts")
    sentence2 = "彼女は手紙を書いた。"
    candidate2 = "書いた"
    print(f"Sentence: {sentence2}")
    print(f"Candidate: {candidate2}")

    doc2 = nlp(sentence2)
    print("Tokenization:")
    for token in doc2:
        if token.pos_ in ['VERB', 'ADJ', 'AUX'] or token.text in ['書い', 'た']:
            print(f"  '{token.text}' → lemma: '{token.lemma_}' (POS: {token.pos_})")

    # Show morphological matching
    print("Morphological matching:")
    for token in doc2:
        if (token.pos_ in ['VERB', 'ADJ'] and
                candidate2.startswith(token.text)):
            print(f"  🎯 STEM MATCH: '{token.text}' → lemma: '{token.lemma_}'")
    print()

    # Example 3: Split into 3+ parts (complex morphology)
    print("🔹 EXAMPLE 3: Split into 3+ parts")
    sentence3 = "昨日は暑くなかった。"
    candidate3 = "暑くなかった"
    print(f"Sentence: {sentence3}")
    print(f"Candidate: {candidate3}")

    doc3 = nlp(sentence3)
    print("Tokenization:")
    for token in doc3:
        if token.pos_ in ['VERB', 'ADJ', 'AUX'] or token.text in ['暑く', 'なかっ', 'た']:
            print(f"  '{token.text}' → lemma: '{token.lemma_}' (POS: {token.pos_})")

    # Show morphological matching
    print("Morphological matching:")
    for token in doc3:
        if (token.pos_ in ['VERB', 'ADJ'] and
                candidate3.startswith(token.text)):
            print(f"  🎯 STEM MATCH: '{token.text}' → lemma: '{token.lemma_}'")
    print()

    # Summary
    print("=== SUMMARY ===")
    print("1. 本 (lemma form) → 本 (no splitting, exact match works)")
    print("2. 書いた (past tense) → 書い + た (2 parts, need stem matching)")
    print("3. 暑くなかった (negative past) → 暑く + なかっ + た (3 parts, need stem matching)")
    print("\nOur algorithm should handle all three cases! 🎯")


if __name__ == "__main__":
    test_ginza_tokenization()
    test_morphological_complexity()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re


def count_analysis(text):
    """
    Analyzes a text and shows different counting methods.
    """
    print(f"Text: '{text}'")
    print(f"├── Characters (len()): {len(text)}")

    # Count words (separated by spaces)
    words_by_space = text.split()
    print(f"├── Words (by spaces): {len(words_by_space)} → {words_by_space}")

    # Count Japanese characters (as in the original method)
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)
    print(f"├── Japanese characters: {len(japanese_chars)} → {japanese_chars}")

    # Non-Japanese characters
    non_japanese_chars = re.findall(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\s]', text)
    print(f"├── Non-Japanese characters: {len(non_japanese_chars)} → {non_japanese_chars}")

    # Ratio of Japanese characters (as in the original method)
    total_chars = len(text)
    if total_chars > 0:
        japanese_ratio = len(japanese_chars) / total_chars
        print(f"└── Japanese character ratio: {japanese_ratio:.2%}")

    print()


def simulate_original_method_logic(text):
    """
    Simulates the logic from the _simple_japanese_sentence_split method
    """
    print(f"=== Simulation of original method for: '{text}' ===")

    # As in the original method
    total_chars = len(text)
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))

    print(f"total_chars = len(sentence_str) = {total_chars}")
    print(f"japanese_chars = len(re.findall(...)) = {japanese_chars}")

    if total_chars > 0:
        ratio = japanese_chars / total_chars
        print(f"Japanese ratio: {japanese_chars}/{total_chars} = {ratio:.3f}")
        print(f"Would text be accepted? (>= 30% Japanese): {ratio >= 0.3}")

    # Length check
    length_ok = 8 <= total_chars <= 300
    print(f"Length OK? (8-300 characters): {length_ok}")
    print()


def main():
    print("=" * 60)
    print("TEST: Are characters or words being counted?")
    print("=" * 60)

    # Test cases
    test_cases = [
        "車",  # 1 Japanese character
        "car",  # 3 English characters
        "車は赤いです",  # 6 Japanese characters
        "The car is red",  # 14 characters (incl. spaces), 4 words
        "車 car 赤い red",  # Mixed: Japanese and English
        "私は東京に住んでいます。",  # Longer Japanese sentence
        "I live in Tokyo and drive a 車.",  # Mixed sentence
        "私はNetflixが好きです"  # Japanese with English brand name
    ]

    print("DETAILED ANALYSIS:")
    print("-" * 40)
    for text in test_cases:
        count_analysis(text)

    print("\nSIMULATION OF ORIGINAL METHOD:")
    print("-" * 50)
    for text in test_cases:
        simulate_original_method_logic(text)

    print("CONCLUSION:")
    print("-" * 20)
    print("✓ len() counts CHARACTERS, not words")
    print("✓ '車' = 1 character")
    print("✓ 'car' = 3 characters")
    print("✓ The original method definitely uses character counting")

    # Additional proof
    print("\nADDITIONAL PROOF:")
    print("-" * 30)
    word_with_spaces = "hello world"
    word_without_spaces = "helloworld"

    print(f"'{word_with_spaces}' has {len(word_with_spaces)} characters (incl. spaces)")
    print(f"'{word_without_spaces}' has {len(word_without_spaces)} characters")
    print(f"Both have {len(word_with_spaces.split())} and {len(word_without_spaces.split())} words respectively")

    print("\n→ len() clearly counts characters, not words!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test file: Validate student implementations of NGramCharLM class in hw3.py

This script trains an N-gram language model on Shakespeare text and tests various functionalities.
Students can run this script to verify their implementation is correct.

Usage:
    python test_ngram.py
"""

import urllib.request
import sys
import os

# Import student implementation
try:
    from hw3 import NGramCharLM
except ImportError:
    print("Error: Cannot import NGramCharLM class from hw3.py")
    print("Please ensure hw3.py is in the same directory and NGramCharLM class is properly implemented")
    sys.exit(1)


def download_shakespeare_text():
    """Download Shakespeare text for training"""
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    print("Downloading Shakespeare corpus...")
    
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8", errors="ignore")
        print(f"Successfully downloaded text, total characters: {len(text)}")
        return text
    except Exception as e:
        print(f"Download failed: {e}")
        print("Using simple test text instead...")
        return "To be, or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune."


def test_ngram_implementation():
    """Test NGramCharLM class implementation"""

    # Download training text
    text = download_shakespeare_text()

    print("\n" + "=" * 60)
    print("Starting NGramCharLM implementation tests")
    print("=" * 60)

    # Test 1: Model initialization and training
    print("\n1. Testing model initialization and training...")
    try:
        model = NGramCharLM(n=10)
        print(f"   ✓ Successfully created 10-gram model")
        print(f"   Initial state - trained: {model.trained}, vocab size: {len(model.vocab)}")

        model.fit(text)
        print(f"   ✓ Training completed")
        print(f"   Post-training state - trained: {model.trained}, vocab size: {len(model.vocab)}")
        print(f"   Number of contexts: {len(model.counts)}")

    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return False

    # Test 2: Probability calculation
    print("\n2. Testing probability calculation...")
    try:
        test_string = "To be, or not to be, that is the question:\n"
        log_prob = model.logprob(test_string)
        prob = model.prob(test_string)

        print(f"   Test string: {repr(test_string)}")
        print(f"   ✓ Log probability: {log_prob}")
        print(f"   ✓ Probability (may underflow): {prob}")

        # Test empty string
        empty_log_prob = model.logprob("")
        print(f"   Empty string log probability: {empty_log_prob}")

    except Exception as e:
        print(f"   ✗ Probability calculation failed: {e}")
        return False

    # Test 3: Next character distribution
    print("\n3. Testing next character probability distribution...")
    try:
        context = "To be, or not to be, "
        distribution = model.next_char_distribution(context)

        print(f"   Context: {repr(context)}")
        print(f"   ✓ Got probability distribution with {len(distribution)} characters")

        print("   Top 10 most likely next characters:")
        count = 0
        for char, prob in distribution.items():
            if count >= 10:
                break
            display_char = char if char != '\n' else '\\n'
            print(f"      {repr(display_char):>6} : {prob:.6f}")
            count += 1

    except Exception as e:
        print(f"   ✗ Next character distribution calculation failed: {e}")
        return False

    # Text generation
    print("\n4. Testing text generation...")
    seed = "Romeo and "
    generated = model.generate(500, seed=seed)

    print(f"   Seed text: {repr(seed)}")
    print(f"   ✓ Successfully generated {len(generated) - len(seed)} new characters")
    print(f"   Generated text length: {len(generated)}")

    print("\n   Generated text sample:")
    print("   " + "-" * 50)
    # Only display first 300 characters to avoid overly long output
    display_text = generated[:300]
    if len(generated) > 300:
        display_text += "..."
    print("   " + display_text.replace('\n', '\n   '))
    print("   " + "-" * 50)

    # Test 4: Edge cases
    print("\n5. Testing edge cases...")
    try:
        # Test different n values
        for n in [1, 2, 3]:
            small_model = NGramCharLM(n=n)
            small_model.fit("hello world")
            small_prob = small_model.logprob("hello")
            print(f"   ✓ {n}-gram model works correctly, 'hello' log probability: {small_prob:.4f}")

    except Exception as e:
        print(f"   ✗ Edge case testing failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 All tests passed! Your NGramCharLM implementation looks correct.")
    print("=" * 60)

    return True


def run_interactive_demo():
    """Run interactive demo"""
    print("\nWould you like to run an interactive demo? (y/n): ", end="")
    try:
        choice = input().lower().strip()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return

    if choice != 'y':
        return

    print("\nStarting interactive demo...")

    # Retrain a smaller model for demo
    demo_text = download_shakespeare_text()
    model = NGramCharLM(n=5)
    model.fit(demo_text)

    print("\nYou can input a context, and the model will show the next character probability distribution")
    print("Type 'quit' to exit the demo")

    while True:
        try:
            print("\nPlease enter context: ", end="")
            context = input()

            if context.lower() == 'quit':
                break

            distribution = model.next_char_distribution(context)

            print(f"\nNext character probability distribution for context '{context}' (top 10):")
            count = 0
            for char, prob in distribution.items():
                if count >= 10:
                    break
                display_char = char if char != '\n' else '\\n'
                print(f"  {repr(display_char):>6} : {prob:.6f}")
                count += 1

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDemo ended, goodbye!")


if __name__ == "__main__":
    print("NGramCharLM Implementation Test Script")
    print("=" * 40)
    
    # Run tests
    success = test_ngram_implementation()
    
    if success:
        # If tests pass, offer interactive demo option
        run_interactive_demo()
    else:
        print("\nPlease fix the above errors and rerun the test.")
        sys.exit(1)
#!/usr/bin/env python3
"""
Sanity check script for tokenizer/model compatibility verification.
Run this to verify your setup before training or inference.

Usage:
    python sanity_check.py
"""

import os
import sys
from src.config import TrainConfig
from src.utils import validate_vocab_size, validate_language_token


def main():
    cfg = TrainConfig()
    
    print("=" * 60)
    print("Chatterbox Turbo - Sanity Check")
    print("=" * 60)
    
    # Check if pretrained_models directory exists
    if not os.path.exists(cfg.model_dir):
        print(f"\nERROR: Model directory '{cfg.model_dir}' not found.")
        print("Please run 'python setup.py' first to download the models.")
        sys.exit(1)
    
    # Try to load the tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model_dir, use_fast=True)
    except Exception as e:
        print(f"\nERROR: Could not load tokenizer from '{cfg.model_dir}'")
        print(f"Error: {e}")
        print("Please run 'python setup.py' first.")
        sys.exit(1)
    
    print(f"\n1. TOKENIZER INFO")
    print(f"   Path: {cfg.model_dir}")
    print(f"   Tokenizer vocab size (len): {len(tok)}")
    print(f"   Config vocab size: {cfg.new_vocab_size}")
    
    # Vocab size check
    print(f"\n2. VOCAB SIZE VALIDATION")
    try:
        validate_vocab_size(tok, cfg.new_vocab_size)
        print("   ✅ PASS: Tokenizer size matches config")
    except ValueError as e:
        print(f"   ❌ FAIL: {e}")
        print("\n   To fix: Update 'new_vocab_size' in src/config.py to match the tokenizer size.")
    
    # Language tag check
    print(f"\n3. LANGUAGE TAG VALIDATION")
    print(f"   Target language: {cfg.target_language}")
    tag = f"[{cfg.target_language}]"
    tag_ids = tok.encode(tag, add_special_tokens=False)
    print(f"   Tag '{tag}' tokenization: {tag_ids}")
    
    if len(tag_ids) == 1:
        print(f"   ✅ PASS: Language tag is a single token (ID: {tag_ids[0]})")
    else:
        print(f"   ⚠️  WARNING: Language tag is split into {len(tag_ids)} tokens")
        print("   This may affect model performance. Consider re-running setup.py.")
    
    # Spanish text tests
    print(f"\n4. SPANISH TEXT TOKENIZATION")
    tests = [
        "Hola, me llamo Sofía y esta es mi voz.",
        "La niña pequeña juega en el jardín con su perro.",
        "¿Cómo estás? ¡Qué alegría verte!",
    ]
    
    for text in tests:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        print(f"\n   TEXT: {text}")
        print(f"   Tokens: {len(ids)}")
        print(f"   First 15 IDs: {ids[:15]}")
    
    # Check for special Spanish characters
    print(f"\n5. SPANISH CHARACTER HANDLING")
    special_chars = ["ñ", "á", "é", "í", "ó", "ú", "ü", "¿", "¡"]
    for char in special_chars:
        char_ids = tok.encode(char, add_special_tokens=False)
        status = "✅" if len(char_ids) <= 2 else "⚠️"
        print(f"   {status} '{char}' -> {char_ids} ({len(char_ids)} token(s))")
    
    # Unicode normalization check
    print(f"\n6. UNICODE NORMALIZATION CHECK")
    from unicodedata import normalize
    
    test_text = "niña"
    nfc = normalize("NFC", test_text)
    nfkd = normalize("NFKD", test_text)
    
    nfc_ids = tok.encode(nfc, add_special_tokens=False)
    nfkd_ids = tok.encode(nfkd, add_special_tokens=False)
    
    print(f"   Original text: '{test_text}'")
    print(f"   NFC normalized: '{nfc}' -> {len(nfc_ids)} tokens")
    print(f"   NFKD normalized: '{nfkd}' -> {len(nfkd_ids)} tokens")
    
    if nfc_ids == nfkd_ids:
        print("   ✅ NFC and NFKD produce same tokenization")
    else:
        print("   ⚠️  NFC and NFKD produce different tokenizations!")
        print("   This repo uses NFC for consistent Spanish handling.")
    
    print("\n" + "=" * 60)
    print("Sanity check complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

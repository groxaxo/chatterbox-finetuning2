import os
import requests
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from src.config import TrainConfig


DEST_DIR = "pretrained_models"
 
# Turbo model files with multilingual support
CHATTERBOX_TURBO_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/ve.safetensors?download=true",
    "t3_turbo_v1.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/t3_turbo_v1.safetensors?download=true",
    "s3gen_meanflow.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen_meanflow.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/conds.pt?download=true",
    "vocab.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/vocab.json?download=true",
    "added_tokens.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/added_tokens.json?download=true",
    "special_tokens_map.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/special_tokens_map.json?download=true",
    "tokenizer_config.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/tokenizer_config.json?download=true",
    "merges.txt": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/merges.txt?download=true",
    "grapheme_mtl_merged_expanded_v1.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}


def download_file(url, dest_path):
    """Downloads a file from a URL to a specific destination with a progress bar."""
    
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading: {os.path.basename(dest_path)}...")
    
    try:
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            
            for data in response.iter_content(block_size):
                
                size = file.write(data)
                bar.update(size)
                
        print(f"Download complete: {dest_path}\n")
        
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)



def merge_and_save_turbo_tokenizer():
    """
    Combines the GPT-2 tokenizer with the multilingual grapheme vocab.
    This enables Spanish and other language support for the Turbo model.
    Also adds language tokens (e.g., [es], [fr]) as single tokens.
    """
    print("\n--- Turbo Vocab Merging for Multilingual Support ---")
    
    try:
        base_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    except Exception as e:
        print(f"ERROR: The original tokenizer could not be loaded. Did you download the files correctly? -> {e}")
        return 0
        
        
    initial_len = len(base_tokenizer)
    print(f"   Original GPT-2 Size: {initial_len}")


    custom_vocab_path = os.path.join(DEST_DIR, "grapheme_mtl_merged_expanded_v1.json")
    
    print(f"Loading: Multilingual Grapheme Vocab ({custom_vocab_path})")
    
    with open(custom_vocab_path, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)


    if "model" in custom_data and "vocab" in custom_data["model"]:
        vocab_dict = custom_data["model"]["vocab"]
        
    else:
        print("Warning: The custom VOCAB format may differ from what is expected.")
        return 0

    unique_tokens_to_add = list(vocab_dict.keys())
    added_count = base_tokenizer.add_tokens(unique_tokens_to_add)
    
    # Add language tokens as single tokens (important for language-specific preprocessing)
    # These must be single tokens for the model to properly learn language conditioning
    language_codes = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", 
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"
    ]
    language_tokens = [f"[{code}]" for code in language_codes]
    
    # Only add tokens that aren't already in the tokenizer
    new_lang_tokens = []
    for token in language_tokens:
        # Check if token is already a single token
        token_ids = base_tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            new_lang_tokens.append(token)
    
    if new_lang_tokens:
        lang_added = base_tokenizer.add_tokens(new_lang_tokens)
        print(f"   Language tokens added: {lang_added} ({new_lang_tokens[:5]}...)")
    else:
        print("   All language tokens already exist as single tokens.")
    
    final_len = len(base_tokenizer)

    print(f"Merging: {added_count} multilingual tokens added (including Spanish characters).")
    print(f"   New Vocab Size: {final_len}")


    print(f"Saving: Writing the combined tokenizer to the '{DEST_DIR}' folder...")
    base_tokenizer.save_pretrained(DEST_DIR)
    
    print("MERGER SUCCESSFUL!")
    
    return final_len



def test_merge_tokenizer_process(tokenizer_path):
    
    try:

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print(f"--- RESULTS ---")
        print(f"Folder: {tokenizer_path}")
        print(f"Actual Vocab Size (len): {len(tok)}")

        # Test Spanish character
        test_token = "[es]"
        test_id = tok.encode(test_token, add_special_tokens=False)
        
        print(f"Test Token '{test_token}' ID: {test_id}")
        
        if len(tok) > 50276:
            print("SUCCESS! Multilingual tokens have been added.")
            
        else:
            print("ERROR: The size still appears old.")


    except Exception as e:
        print(f"Error: {e}")




def main():
    
    cfg = TrainConfig()
    
    print("--- Chatterbox Turbo Setup for Spanish Finetuning ---\n")
    print(f"Target Language: {cfg.target_language}")
    
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
        
    else:
        print(f"Directory found: {DEST_DIR}")

    print(f"Mode: CHATTERBOX-TURBO (Checking {len(CHATTERBOX_TURBO_FILES)} files)")

    # 2. Download files
    for filename, url in CHATTERBOX_TURBO_FILES.items():
        dest_path = os.path.join(DEST_DIR, filename)
        download_file(url, dest_path)

    # 3. Merge tokenizers for multilingual support
    new_vocab_size = merge_and_save_turbo_tokenizer()
    if new_vocab_size > 0:
        
        print("\n" + "="*60)
        print("INSTALLATION COMPLETE (CHATTERBOX-TURBO MODE)")
        print("All models are set up in 'pretrained_models/' folder.")
        print(f"Please update the 'new_vocab_size' value in the 'src/config.py' file")
        print(f"to: {new_vocab_size}")
        print("="*60 + "\n")



if __name__ == "__main__":
    main()
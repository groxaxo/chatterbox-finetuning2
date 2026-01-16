from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the files
    model_dir: str = "./pretrained_models"
    
    # Path to your metadata CSV (Format: ID|RawText|NormText)
    csv_path: str = "./MyTTSDataset/metadata.csv"
    metadata_path: str = "./metadata.json"
    
    # Directory containing WAV files
    wav_dir: str = "./MyTTSDataset/wavs"
    #wav_dir: str = "./FileBasedDataset"
    
    preprocessed_dir = "./MyTTSDataset/preprocess"
    #preprocessed_dir = "./FileBasedDataset/preprocess"
    
    # Output directory for the finetuned model
    output_dir: str = "./chatterbox_output"
    
    is_inference = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Hola, desarrollar mi voz me tomó bastante tiempo y ahora que la tengo, no voy a quedarme callado."

    ljspeech = True # Set True if the dataset format is ljspeech, and False if it's file-based.
    json_format = False # Set True if the dataset format is json, and False if it's file-based or ljspeech.
    preprocess = True # If you've already done preprocessing once, set it to false.

    # --- Language Settings ---
    # Target language for finetuning (Spanish by default)
    # Supported: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
    target_language: str = "es"

    # --- Vocabulary ---
    # The size of the NEW vocabulary (from tokenizer.json)
    # For Turbo mode with multilingual support: Use the exact number provided by setup.py (e.g., 52260)
    new_vocab_size: int = 52260

    # --- Hyperparameters ---
    batch_size: int = 4         # Adjust based on VRAM (2, 4, 8)
    grad_accum: int = 2        # Effective Batch Size = Batch * Accum
    learning_rate: float = 1e-5 # T3 is sensitive, keep low
    num_epochs: int = 100
    
    save_steps: int = 1000
    save_total_limit: int = 2
    dataloader_num_workers: int = 4

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850   # Truncates very long audio
    prompt_duration: float = 3.0 # Duration for the reference prompt (seconds)

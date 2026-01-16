# Chatterbox Turbo: Spanish Finetuning Kit 🎙️🇪🇸

> ## 🚀 **Focused on Chatterbox Turbo with Spanish Language Support** 🚀
>
> This repository is optimized for fine-tuning the **Chatterbox Turbo** model for **Spanish** language TTS.
>
> *   **Architecture:** GPT-2 based Turbo model with fast, high-quality speech synthesis.
> *   **Multilingual Tokenizer:** The setup script **automatically merges** Turbo's large English vocabulary with the 23-language grapheme set, enabling proper Spanish character handling.
> *   **Language-Aware Processing:** Text is preprocessed with NFKD normalization and language tokens (e.g., `[es]`) for optimal Spanish synthesis.


---

A specialized infrastructure for **fine-tuning** the **Chatterbox Turbo** model with your own Spanish dataset and generating high-quality Spanish speech synthesis.

This kit intelligently extends the model's vocabulary with multilingual graphemes for maximum performance on Spanish and other languages.

---


## ⚠️ CRITICAL INFORMATION (Please Read)

### 0. Preprocessing is Mandatory
This repository uses an **offline preprocessing** strategy to maximize training speed. The preprocessing script processes all audio files, extracts speaker embeddings and acoustic tokens, and saves them as `.pt` files.

### 1. Tokenizer and Vocab Size (Most Important)
The Turbo model uses a BPE-based tokenizer that is **automatically extended** with multilingual graphemes during setup.

*   **Default Support:** The merged tokenizer includes characters for Spanish (`ñ, á, é, í, ó, ú, ü, ¿, ¡`) and 22 other languages.
*   **Language Token:** Text is automatically prefixed with `[es]` for Spanish language processing.
*   **Critical:** The `new_vocab_size` variable in `src/config.py` **must exactly match** the value output by `setup.py` (typically 52260).

### 2. Audio Sample Rates
*   **Training (Input):** Chatterbox's encoder and T3 module work with **16,000 Hz (16kHz)** audio. The preprocessing scripts automatically resample to 16kHz.
*   **Output (Inference):** The model's vocoder generates audio at **24,000 Hz (24kHz)**.

---

## 📂 Folder Structure

```text
chatterbox-finetuning/
├── pretrained_models/                             # setup.py downloads required models here
│   ├── ve.safetensors
│   ├── s3gen_meanflow.safetensors
│   ├── t3_turbo_v1.safetensors
│   ├── vocab.json
│   ├── merges.txt
│   └── grapheme_mtl_merged_expanded_v1.json
├── MyTTSDataset/                                  # Your custom dataset in LJSpeech format
│   ├── metadata.csv                               # Dataset metadata (file|text|normalized_text)
│   └── wavs/                                      # Directory containing WAV files
├── speaker_reference/                             # Speaker reference audio files
│   └── reference.wav                              # Reference audio for voice cloning
├── src/
│   ├── config.py                                  # All settings and hyperparameters
│   ├── dataset.py                                 # Data loading and processing
│   ├── model.py                                   # Model weight transfer and training wrapper
│   ├── preprocess_ljspeech.py                     # Preprocessing script for LJSpeech format
│   ├── preprocess_file_based.py                   # Preprocessing script for file-based format
│   ├── preprocess_json.py                         # Preprocessing script for JSON format
│   └── utils.py                                   # Logger and VAD utilities
├── train.py                                       # Main training script
├── inference.py                                   # Speech synthesis script
├── setup.py                                       # Setup script for downloading models
├── requirements.txt                               # Required dependencies
└── README.md                                      # This file
```

---

## 🚀 Installation

### 1. Install Dependencies
Requires Python 3.8+ and GPU (recommended):

**Install FFmpeg (Required):**
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on macOS using Homebrew
brew install ffmpeg

# on Windows using Chocolatey
choco install ffmpeg
```

**Install Python Dependencies:**
```bash
git clone https://github.com/groxaxo/chatterbox-finetuning2.git
cd chatterbox-finetuning2

pip install -r requirements.txt
```


### 2. Download & Prepare Models

**Step 2.1: Configure Target Language**
Open `src/config.py` and set your target language (default is Spanish):

```python
# In src/config.py
target_language: str = "es"  # Spanish
```

**Step 2.2: Run the Setup Script**
This command downloads the Turbo model files and merges the tokenizers for multilingual support:

```bash
python setup.py
```

**Step 2.3: Update Vocab Size**
The setup script will output a message like:
`Please update the 'new_vocab_size' in 'src/config.py' to the following value: 52260`

Update the `new_vocab_size` in `src/config.py` to match this value.

---


## 🏋️ Training (Fine-Tuning)

### 1. Dataset Preparation

Your dataset should follow the LJSpeech format with a CSV file:
`filename|raw_text|normalized_text`

Example `metadata.csv` for Spanish:
```text
recording_001|Hola mundo.|hola mundo
recording_002|Esta es una prueba de grabación.|esta es una prueba de grabación
```

Place your dataset in the `MyTTSDataset/` folder:
```text
MyTTSDataset/
├── metadata.csv
└── wavs/
    ├── recording_001.wav
    ├── recording_002.wav
    └── ...
```

**Dataset Quality Requirements:**
- Sample rate: 16kHz, 22.05kHz, or 44.1kHz (will be resampled to 16kHz automatically)
- Format: WAV (mono or stereo - will be converted to mono automatically)
- Duration: 3-10 seconds per segment (optimal for TTS)
- Minimum total duration: 30+ minutes for basic training
- **Recommended:** 1 hour of clean Spanish audio for optimal results

### 2. Configuration
Adjust key settings in `src/config.py`:

```python
# Language settings
target_language: str = "es"  # Spanish

# Vocabulary (must match setup.py output)
new_vocab_size: int = 52260

# Training hyperparameters
batch_size: int = 4         # Adjust based on your GPU VRAM
learning_rate: float = 1e-5
num_epochs: int = 100
```

If your dataset is file-based (not LJSpeech format), set:
```python
ljspeech = False
```

### 3. Start Training
```bash
python train.py
```

The trained model will be saved as `chatterbox_output/t3_turbo_finetuned.safetensors`.

**Training Tips:**
*   **VRAM:** For 12GB VRAM, use `batch_size=4`. For lower VRAM, use `batch_size=2` with `grad_accum=32`.
*   **Mixed Precision:** Uses `bf16=True` by default for faster training and memory efficiency.
*   **Recommended Duration:** For optimal results with 1 hour of Spanish audio, train for **100-150 epochs**.

---

## 🗣️ Inference (Speech Synthesis)

### 1. Prepare Reference Audio
Place your reference audio in `speaker_reference/`:
```text
speaker_reference/
└── reference.wav
```

**Reference Audio Requirements:**
*   Format: WAV, mono or stereo
*   Duration: 5-10 seconds recommended
*   Quality: Clean audio with minimal background noise

### 2. Running Inference
Edit `inference.py` to set your Spanish text:

```python
TEXT_TO_SAY = "Hola, esta es una prueba del modelo afinado para español."
AUDIO_PROMPT = "./speaker_reference/reference.wav"
```

Run inference:
```bash
python inference.py
```

The output will be saved as `output.wav` (24kHz).

---

## 🌐 Supported Languages

The tokenizer supports 23 languages. Set `target_language` in `src/config.py`:

| Code | Language | Code | Language |
|------|----------|------|----------|
| ar | Arabic | ms | Malay |
| da | Danish | nl | Dutch |
| de | German | no | Norwegian |
| el | Greek | pl | Polish |
| en | English | pt | Portuguese |
| **es** | **Spanish** | ru | Russian |
| fi | Finnish | sv | Swedish |
| fr | French | sw | Swahili |
| he | Hebrew | tr | Turkish |
| hi | Hindi | zh | Chinese |
| it | Italian | ko | Korean |
| ja | Japanese | | |

---

## 🛠️ Technical Details

### Tokenizer Structure (Turbo Model)
The Turbo model uses GPT-2's powerful BPE tokenizer as a base. The `setup.py` script performs **Vocab Extension**:
1. Starts with GPT-2 vocabulary (~50,000 tokens)
2. Adds multilingual grapheme tokens (Spanish characters like `ñ, á, é, í, ó, ú`)
3. Adds language tokens like `[es]` for language identification

### Language-Aware Text Processing
For Spanish and other languages:
1. Text is normalized using NFKD Unicode normalization
2. Language token `[es]` is prepended to the text
3. Special Spanish punctuation (¿, ¡) is preserved

### VAD Integration
During inference, Silero VAD automatically trims unwanted silence and noise from generated audio.

---

## 📝 Troubleshooting

**Error:** `RuntimeError: Error(s) in loading state_dict for T3... size mismatch`
*   **Solution:** `new_vocab_size` doesn't match the tokenizer. Update it to match the value from `setup.py`.

**Error:** `FileNotFoundError: ... ve.safetensors`
*   **Solution:** Run `python setup.py` to download base models.

**Error:** `CUDA out of memory`
*   **Solution:** Reduce `batch_size` in `src/config.py` or use gradient accumulation.

**Poor Quality Spanish Output:**
*   Ensure your Spanish training data is clean and properly transcribed
*   Check that `target_language` is set to `"es"`
*   Verify the reference audio is at least 5 seconds long

---

## 🙏 Acknowledgments

Based on the Chatterbox TTS model architecture by ResembleAI. Special thanks to the original authors and contributors.

---

## 📧 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review `src/config.py` for configuration options
3. Open an issue on GitHub with detailed error messages

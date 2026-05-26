import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
# --- Added imports ---
import librosa


# --- 0. Configuration ---

# Please update this path to the absolute path where your AudioSet 'data' folder is located.
AUDIOSET_BASE_DIR = "/mnt/data/AudioSet/data"

# List of audio/ subdirectories to process
AUDIO_SUBDIRS = ['bal_train', 'eval', 'eval_small', 'unbal_train']

# Output file name
OUTPUT_FILE = "audioset_description.jsonl"

# GPU batch size
BATCH_SIZE = 64

# Added: DataLoader configuration
# Adjust according to your number of CPU cores; recommended to set to about half of os.cpu_count()
NUM_WORKERS = 16
# If your GPU is available and the system is Linux, enabling this is recommended to speed up data transfer
PIN_MEMORY = True


# --- 1. Custom Dataset ---
class AudioSetDataset(Dataset):
    """
    Creates a PyTorch Dataset for AudioSet.
    It is only responsible for returning the audio file path and metadata given an index.
    The actual file loading and processing happens in collate_fn to fully leverage multiprocessing.
    """

    def __init__(self, audio_files, segments_df, ontology_map, user_prompt):
        self.audio_files = audio_files
        self.segments_df = segments_df
        self.ontology_map = ontology_map
        self.user_prompt = user_prompt

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio_path_str = str(audio_path)
        audio_filename = audio_path.name
        ytid = audio_path.stem

        # --- Metadata lookup logic ---
        try:
            segment_info = self.segments_df.loc[ytid]
            label_ids_str = segment_info['positive_labels']
            label_ids = label_ids_str.split(',')
            name_parts = [self.ontology_map.get(label_id.strip()) for label_id in label_ids]
            event_name = ", ".join(filter(None, name_parts)) if name_parts else "Unknown"
        except KeyError:
            event_name = "Metadata Not Found"
        except Exception as e:
            event_name = f"Error: {str(e)}"
        # --- End ---

        metadata = {
            "audio_filename": audio_filename,
            "event": event_name
        }

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": self.user_prompt},
                {"type": "audio", "path": audio_path_str},
            ]},
        ]

        return metadata, messages


# --- 2. Custom Collate Function (modified) ---
class Collator:
    """
    A callable class used as the DataLoader's collate_fn.
    It packs a batch of samples obtained from the Dataset (metadata and messages)
    into the tensor format required by the model.

    Optimizations:
    1. Load audio manually inside the Collator and use librosa's `duration` parameter to limit the loaded duration,
       fundamentally avoiding out-of-memory errors caused by loading overly long files.
    2. Add a try-except block to catch and handle corrupted or unreadable audio files,
       allowing the program to skip bad samples and continue running, improving robustness.
    """
    def __init__(self, processor, max_duration_seconds=10):
        self.processor = processor
        self.target_sampling_rate = processor.feature_extractor.sampling_rate
        self.max_duration_seconds = max_duration_seconds
        print(f"Collator initialized: only the first {max_duration_seconds} seconds of audio will be loaded.")

    def __call__(self, batch):
        # Unzip the batch data
        list_metadata, list_messages = zip(*batch)

        valid_metadata = []
        valid_messages = []

        # Process audio files one by one instead of handing the entire list to the processor
        for metadata, messages in zip(list_metadata, list_messages):
            audio_path = messages[0]['content'][1]['path']
            try:
                # [Key change] Use librosa to load directly and limit the duration
                audio_array, _ = librosa.load(
                    audio_path,
                    sr=self.target_sampling_rate,
                    duration=self.max_duration_seconds
                )

                # Replace the original path with the loaded audio array
                messages[0]['content'][1] = {
                    "type": "audio",
                    "audio": audio_array,
                    "sampling_rate": self.target_sampling_rate
                }
                valid_metadata.append(metadata)
                valid_messages.append(messages)

            except Exception as e:
                # If a file fails to load (e.g., the file is corrupted), print the error and skip it
                tqdm.write(f"Warning: failed to load audio file, skipped. File: {audio_path}, error: {e}")
                continue

        # If all files in the entire batch failed to load, return None
        if not valid_messages:
            return None, None

        # Use the processor to process the entire valid batch
        model_inputs = self.processor.apply_chat_template(
            list(valid_messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return list(valid_metadata), model_inputs


# --- 3. Metadata loading function ---
def load_audioset_metadata(base_dir):
    """
    Load AudioSet metadata from ontology.json and the segments CSV files.
    """
    print("Loading AudioSet metadata...")
    base_path = Path(base_dir)

    ontology_path = base_path / "ontology.json"
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)
        ontology_map = {item['id']: item['name'] for item in ontology_data}
        print(f"Ontology file loaded successfully: {ontology_path}")
    except Exception as e:
        print(f"Failed to load ontology file: {ontology_path}, error: {e}")
        raise

    segment_files = [
        "unbalanced_train_segments.csv",
        "balanced_train_segments.csv",
        "eval_segments.csv"
    ]
    all_dfs = []
    for fname in segment_files:
        csv_path = base_path / fname
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.DtypeWarning)
                df = pd.read_csv(
                    csv_path, header=None, comment='#', quotechar='"',
                    skipinitialspace=True,
                    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels']
                )
            df.set_index('YTID', inplace=True)
            all_dfs.append(df)
            print(f"Segments CSV file loaded successfully: {csv_path}")
        except Exception as e:
            print(f"Failed to load CSV file: {csv_path}, error: {e}")
            print(f"Warning: unable to load {csv_path}, processing will continue.")

    if not all_dfs:
        raise ValueError("Failed to load any segments CSV file; cannot continue.")

    combined_segments_df = pd.concat(all_dfs)
    combined_segments_df = combined_segments_df[~combined_segments_df.index.duplicated(keep='first')]
    print(f"Metadata loaded; contains {len(combined_segments_df)} unique audio segment records in total.")

    return ontology_map, combined_segments_df


# --- 4. Main function ---
def main():
    # --- Load metadata ---
    try:
        ontology_map, segments_df = load_audioset_metadata(AUDIOSET_BASE_DIR)
    except Exception as e:
        print(f"Failed to initialize metadata: {e}")
        return

    # --- Load the model ---
    model_id = "mispeech/midashenglm-7b"
    print("\n--- Initializing model, tokenizer, and processor ---")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        print(f"Loading model from '{model_id}'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error while loading the model: {e}")
        return

    # --- Prepare the prompt template and file list ---
    user_prompt = (
        "Craft a rich and vivid narrative describing the audio provided. Your description should be a comprehensive auditory scene analysis, identifying every sound present—from spoken words and musical scores to subtle background noises. "
        "Detail the specific characteristics of each sound (e.g., pitch, tempo, volume, timbre) and use this information to paint a picture of the environment, context, and overall mood or atmosphere of the recording."
    )

    audio_files_to_process = []
    audio_root_path = Path(AUDIOSET_BASE_DIR) / "audio"
    for subdir in AUDIO_SUBDIRS:
        current_dir = audio_root_path / subdir
        if current_dir.exists():
            print(f"Scanning folder: {current_dir}")
            audio_files_to_process.extend(list(current_dir.rglob("*.flac")))
        else:
            print(f"Warning: folder does not exist, skipped: {current_dir}")

    print(f"\nScan complete; found {len(audio_files_to_process)} .flac audio files in total.")

    # --- Use Dataset and DataLoader ---
    # 1. Instantiate the Dataset
    dataset = AudioSetDataset(audio_files_to_process, segments_df, ontology_map, user_prompt)

    # 2. Instantiate the Collator
    collator = Collator(processor, max_duration_seconds=10)  # Limit audio to a maximum of 10 seconds

    # 3. Instantiate the DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Shuffling is usually not needed during inference
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=PIN_MEMORY,
        drop_last=False  # Process all files, even if the last batch has fewer than BATCH_SIZE
    )

    print(f"Loading data with the DataLoader, number of worker processes: {NUM_WORKERS}, pinned memory: {PIN_MEMORY}")

    # --- New processing loop ---
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        # Iterate over the DataLoader, which preloads and processes data in the background
        for batch_metadata, model_inputs in tqdm(data_loader, desc="Processing batches"):
            # [Key change] Check whether the Collator returned an empty batch
            if batch_metadata is None or model_inputs is None:
                tqdm.write("Warning: skipping a completely invalid batch.")
                continue

            # Move the data to the device where the model resides
            model_inputs = model_inputs.to(model.device)

            # b. [GPU inference]
            try:
                with torch.no_grad():
                    inputs = {
                        key: value.to(model.dtype) if torch.is_floating_point(value) else value
                        for key, value in model_inputs.items()
                    }
                    # Run inference using the new dict with converted data types
                    generation = model.generate(**inputs, max_new_tokens=256)
                    output_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            except Exception as e:
                tqdm.write(f"GPU inference error while processing the batch: {e}")
                output_texts = [f"Inference Error: {str(e)}"] * len(batch_metadata)

            # c. [CPU serial] Process and write the batch results
            for metadata, full_response in zip(batch_metadata, output_texts):
                # Extract the assistant's reply from the model's full output
                inst_parts = full_response.split('[/INST]', 1)
                if len(inst_parts) > 1:
                    response_text = inst_parts[1].strip()
                else:
                    # As a fallback, if the end marker is not found, return the full content
                    response_text = full_response

                result_record = {
                    "audio_filename": metadata["audio_filename"],
                    "event": metadata["event"],
                    "description": response_text
                }
                f.write(json.dumps(result_record, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete! All descriptions have been saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    # In a multiprocessing environment, disable huggingface tokenizers parallelism to avoid conflicts
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
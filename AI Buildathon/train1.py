
import os
import sys
import torch
import librosa
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import traceback

DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-5

try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f" CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = "cpu"
        print("CUDA not available, using CPU")
except Exception as e:
    DEVICE = "cpu"
    print(f" Error checking CUDA: {e}. Using CPU.")

print("=" * 50)
print("DEEPFAKE AUDIO DETECTION - TRAINING")
print("=" * 50)
print(f"Device          : {DEVICE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning rate   : {LR}")
print(f"Sample rate     : {SAMPLE_RATE} Hz")
print("=" * 50)

def validate_dataset_structure():
    """Validate that the dataset structure exists and is correct."""
    if not os.path.exists(DATASET_PATH):
        print(f"\n ERROR: Dataset path not found: {DATASET_PATH}")
        return False
    
    required_dirs = [
        os.path.join(DATASET_PATH, "train", "real"),
        os.path.join(DATASET_PATH, "train", "fake"),
        os.path.join(DATASET_PATH, "val", "real"),
        os.path.join(DATASET_PATH, "val", "fake")
    ]
    
    missing = [d for d in required_dirs if not os.path.exists(d)]
    if missing:
        print("\n ERROR: Missing required directories:")
        for d in missing:
            print(f"   - {d}")
        return False
    
    return True


def load_files(split):
    """Load audio files with comprehensive error handling."""
    real_dir = os.path.join(DATASET_PATH, split, "real")
    fake_dir = os.path.join(DATASET_PATH, split, "fake")

    real_files = []
    fake_files = []

    try:
      
        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        else:
            print(f" Warning: {split}/real directory not found")
        
        
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        else:
            print(f" Warning: {split}/fake directory not found")
            
    except PermissionError as e:
        print(f" ERROR: Permission denied accessing {split} directory: {e}")
        return [], []
    except OSError as e:
        print(f" ERROR: OS error accessing {split} directory: {e}")
        return [], []
    except Exception as e:
        print(f" ERROR: Failed to load {split} files: {e}")
        traceback.print_exc()
        return [], []

    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f"\n{split.upper()} set:")
    print(f"  Real samples: {len(real_files)}")
    print(f"  Fake samples: {len(fake_files)}")
    print(f"  Total: {len(files)}")

    return files, labels


if not validate_dataset_structure():
    print("\n Dataset validation failed!")
    print("Please ensure your dataset structure is:")
    print("  dataset/")
    print("    train/")
    print("      real/  (put real audio .wav files here)")
    print("      fake/  (put fake audio .wav files here)")
    print("    val/")
    print("      real/")
    print("      fake/")
    sys.exit(1)

train_files, train_labels = load_files("train")
val_files, val_labels = load_files("val")

if len(train_files) == 0:
    print("\nERROR: No training files found!")
    print("Please ensure your dataset structure is:")
    print("  dataset/")
    print("    train/")
    print("      real/  (put real audio .wav files here)")
    print("      fake/  (put fake audio .wav files here)")
    print("    val/")
    print("      real/")
    print("      fake/")
    sys.exit(1)

if len(val_files) == 0:
    print("\n ERROR: No validation files found!")
    print("Training requires validation data for model selection.")
    sys.exit(1)


class AudioDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.failed_files = set()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            if not os.path.exists(self.files[idx]):
                raise FileNotFoundError(f"File not found: {self.files[idx]}")
            
            audio, _ = librosa.load(self.files[idx], sr=SAMPLE_RATE)
            
            if len(audio) == 0:
                raise ValueError(f"Empty audio file: {self.files[idx]}")
            
            return audio, self.labels[idx]
            
        except FileNotFoundError as e:
            if self.files[idx] not in self.failed_files:
                print(f"\n File not found: {self.files[idx]}")
                self.failed_files.add(self.files[idx])
            return np.zeros(SAMPLE_RATE, dtype=np.float32), self.labels[idx]
            
        except Exception as e:
            if self.files[idx] not in self.failed_files:
                print(f"\n Error loading {self.files[idx]}: {e}")
                self.failed_files.add(self.files[idx])
            return np.zeros(SAMPLE_RATE, dtype=np.float32), self.labels[idx]


print("\n Loading Wav2Vec2 processor...")
try:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print(" Processor loaded successfully")
except Exception as e:
    print(f" ERROR: Failed to load Wav2Vec2 processor: {e}")
    print("   Please check your internet connection and try again.")
    sys.exit(1)

def collate_fn(batch):
    try:
        audios, labels = zip(*batch)

        inputs = processor(
            list(audios),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        return inputs.input_values, torch.tensor(labels)
    except Exception as e:
        print(f" ERROR in collate_fn: {e}")
        raise

train_loader = DataLoader(
    AudioDataset(train_files, train_labels),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    AudioDataset(val_files, val_labels),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        try:
            features = self.encoder(x).last_hidden_state
            pooled = features.mean(dim=1)
            return torch.sigmoid(self.classifier(pooled))
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}")

print(" Loading Wav2Vec2 model...")
try:
    model = DeepfakeDetector().to(DEVICE)
    print(f" Model loaded on {DEVICE}")
except Exception as e:
    print(f" ERROR: Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("\n" + "=" * 50)
print(" STARTING TRAINING")
print("=" * 50)

best_acc = 0.0
MODEL_SAVE_DIR = "model"

try:
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
except PermissionError as e:
    print(f" ERROR: Permission denied creating {MODEL_SAVE_DIR}: {e}")
    MODEL_SAVE_DIR = "."  
except Exception as e:
    print(f"Warning: Could not create model directory: {e}")
    MODEL_SAVE_DIR = "."

for epoch in range(EPOCHS):
    try:
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x, y in progress_bar:
            try:
                x = x.to(DEVICE)
                y = y.float().to(DEVICE).unsqueeze(1)

                preds = model(x)
                loss = criterion(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                pred_labels = (preds > 0.5).int().squeeze()
                train_correct += (pred_labels == y.squeeze().int()).sum().item()
                train_total += y.size(0)
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nGPU out of memory! Clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                raise

        avg_loss = total_loss / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1) * 100

        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                try:
                    x = x.to(DEVICE)
                    y_float = y.float().to(DEVICE).unsqueeze(1)
                    y = y.to(DEVICE)

                    preds = model(x)
                    val_loss += criterion(preds, y_float).item()
                    
                    preds = (preds > 0.5).int().squeeze()
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                except Exception as e:
                    print(f"\n Validation batch error: {e}")
                    continue

        val_acc = correct / max(total, 1) * 100
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"\n{'='*50}")
        print(f" Epoch {epoch+1}/{EPOCHS} Results")
        print(f"{'='*50}")
        print(f"Train Loss     : {avg_loss:.4f}")
        print(f"Train Accuracy : {train_acc:.2f}%")
        print(f"Val Loss       : {avg_val_loss:.4f}")
        print(f"Val Accuracy   : {val_acc:.2f}%")
        
       
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(MODEL_SAVE_DIR, "deepfake_model_best.pth")
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved! (Accuracy: {val_acc:.2f}%)")
            except Exception as e:
                print(f" ERROR: Failed to save best model: {e}")
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        try:
            interrupt_path = os.path.join(MODEL_SAVE_DIR, "deepfake_model_interrupted.pth")
            torch.save(model.state_dict(), interrupt_path)
            print(f" Model saved to {interrupt_path}")
        except Exception as e:
            print(f" Failed to save model: {e}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n ERROR in epoch {epoch+1}: {e}")
        traceback.print_exc()
        continue


final_model_path = os.path.join(MODEL_SAVE_DIR, "deepfake_model.pth")
try:
    torch.save(model.state_dict(), final_model_path)
    print(f"\n Final model saved to: {final_model_path}")
except PermissionError as e:
    print(f"\n ERROR: Permission denied saving model: {e}")
except IOError as e:
    print(f"\n ERROR: I/O error saving model: {e}")
except Exception as e:
    print(f"\n ERROR: Failed to save final model: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print(" TRAINING COMPLETE!")
print("=" * 50)
print(f"Best Validation Accuracy: {best_acc:.2f}%")
print(f"Final model saved as: {final_model_path}")
print(f"Best model saved as : {os.path.join(MODEL_SAVE_DIR, 'deepfake_model_best.pth')}")
print("=" * 50)

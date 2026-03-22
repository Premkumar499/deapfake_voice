
import os
import sys
import random
import torch
import librosa
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import traceback


import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
OLD_MODEL = os.path.join(SCRIPT_DIR, "deepfake_project/deepfake_model_best.pth")

SAMPLE_RATE = 16000
MAX_LEN = SAMPLE_RATE * 3      
BATCH_SIZE = 11
EPOCHS = 8
LR = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"  
print("\n" + "="*50)
print("  DEVICE CONFIGURATION")
print("="*50)
print(f"   Device: {DEVICE.upper()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True  
    print("   cuDNN Benchmark: Enabled")
else:
    print("    No GPU detected — training will run on CPU (much slower)")
print(f"   PyTorch Version: {torch.__version__}")


def validate_dataset_path():
    """Validate that dataset path exists and has required structure."""
    if not os.path.exists(DATASET_PATH):
        print(f" ERROR: Dataset path not found: {DATASET_PATH}")
        print("   Please ensure the 'dataset' folder exists in the project directory.")
        return False
    
    required_dirs = [
        os.path.join(DATASET_PATH, "train", "real"),
        os.path.join(DATASET_PATH, "train", "fake"),
        os.path.join(DATASET_PATH, "val", "real"),
        os.path.join(DATASET_PATH, "val", "fake")
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    if missing_dirs:
        print(" ERROR: Missing required dataset directories:")
        for d in missing_dirs:
            print(f"   - {d}")
        return False
    
    print(" Dataset structure validated successfully!")
    return True


def load_files(split):
    """Load audio files from dataset split with error handling."""
    real = []
    fake = []

    rdir = os.path.join(DATASET_PATH, split, "real")
    fdir = os.path.join(DATASET_PATH, split, "fake")

    try:
        if os.path.exists(rdir):
            real = [os.path.join(rdir, f) for f in os.listdir(rdir) if f.endswith(('.wav', '.mp3', '.flac'))]
        else:
            print(f" Warning: {split}/real directory not found")

        if os.path.exists(fdir):
            fake = [os.path.join(fdir, f) for f in os.listdir(fdir) if f.endswith(('.wav', '.mp3', '.flac'))]
        else:
            print(f" Warning: {split}/fake directory not found")
            
    except PermissionError as e:
        print(f" ERROR: Permission denied accessing {split} directory: {e}")
        return [], []
    except Exception as e:
        print(f" ERROR: Failed to load {split} files: {e}")
        return [], []

    print(f"{split}: Real={len(real)} Fake={len(fake)}")
    
    if len(real) == 0 and len(fake) == 0:
        print(f" Warning: No audio files found in {split} split!")

    return real + fake, [0] * len(real) + [1] * len(fake)


print("\n" + "="*50)
print(" VALIDATING DATASET...")
print("="*50)

if not validate_dataset_path():
    print("\n Dataset validation failed. Exiting...")
    sys.exit(1)

train_files, train_labels = load_files("train")
val_files, val_labels = load_files("val")

# Check if we have enough data to train
if len(train_files) == 0:
    print(" ERROR: No training files found! Cannot proceed with training.")
    sys.exit(1)

if len(val_files) == 0:
    print(" ERROR: No validation files found! Cannot proceed with training.")
    sys.exit(1)

print(f"\n Dataset loaded successfully!")
print(f"   Total training samples: {len(train_files)}")
print(f"   Total validation samples: {len(val_files)}")


def augment(audio):
    
    if random.random() < 0.3:
        noise = np.random.randn(len(audio)) * 0.003
        audio = audio + noise

   
    if random.random() < 0.3:
        rate = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)

    
    if random.random() < 0.3:
        steps = random.randint(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)

    return audio


class AudioDataset(Dataset):
    def __init__(self, files, labels, train=True):
        self.files = files
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            audio, _ = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        except FileNotFoundError:
            print(f" File not found: {self.files[idx]}")
            
            audio = np.zeros(MAX_LEN, dtype=np.float32)
        except Exception as e:
            print(f" Error loading audio file {self.files[idx]}: {e}")
            audio = np.zeros(MAX_LEN, dtype=np.float32)

        try:
            if self.train:
                audio = augment(audio)
        except Exception as e:
            print(f" Augmentation failed, using original audio: {e}")

        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        else:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))

        return audio.astype(np.float32), self.labels[idx]


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


def collate_fn(batch):
    audios, labels = zip(*batch)

    inputs = processor(
        list(audios),
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt"
    )

    return inputs.input_values, torch.tensor(labels)


train_loader = DataLoader(
    AudioDataset(train_files, train_labels, True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=(DEVICE == "cuda"),  
    num_workers=2 if DEVICE == "cuda" else 0  
)

val_loader = DataLoader(
    AudioDataset(val_files, val_labels, False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=(DEVICE == "cuda"),
    num_workers=2 if DEVICE == "cuda" else 0
)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state
        x = x.mean(dim=1)  
        return self.classifier(x)


model = Detector().to(DEVICE)

print("\n" + "="*50)
print(" LOADING MODEL...")
print("="*50)

model_loaded = False
if os.path.exists(OLD_MODEL):
    print(f"Found old model at: {OLD_MODEL}")
    try:
        state_dict = torch.load(OLD_MODEL, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model_loaded = True
        print(" Old model loaded successfully!")
    except RuntimeError as e:
        print(f" Warning: Model architecture mismatch: {e}")
        print("   Training from scratch with new model...")
    except Exception as e:
        print(f" Warning: Failed to load old model: {e}")
        print("   Training from scratch...")
else:
    print(f" No old model found at: {OLD_MODEL}")
    print("   Training from scratch.")

if model_loaded:
    print("   Model state: Pre-trained weights loaded")
else:
    print("   Model state: Initialized with Wav2Vec2 base weights")


print(" Freezing encoder layers...")
for p in model.encoder.parameters():
    p.requires_grad = False


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)  # Only active on GPU
best_acc = 0


print(f"\n Starting training on {DEVICE.upper()}")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LR}")
print(f"   Train Samples: {len(train_files)}")
print(f"   Val Samples: {len(val_files)}")
print("-" * 50)

for epoch in range(EPOCHS):

   
    if epoch == 3:
        print("\n Unfreezing last 2 encoder layers...")
        for p in model.encoder.encoder.layers[-2:].parameters():
            p.requires_grad = True
        
       
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR * 0.5  
        )
        print("   Optimizer updated with new parameters")

    model.train()
    correct = 0
    total = 0
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for x, y in loop:
        x = x.to(DEVICE, non_blocking=True)  
        y = y.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  
      
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        preds = (torch.sigmoid(logits) > 0.45).int()
        correct += (preds == y.int()).sum().item()
        total += y.size(0)
        train_loss += loss.item()

        loop.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = correct / total * 100
    avg_train_loss = train_loss / len(train_loader)

    
    model.eval()
    v_correct = 0
    v_total = 0
    v_loss = 0

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y.float().unsqueeze(1))
            
            preds = (torch.sigmoid(logits) > 0.45).int().squeeze()
            v_correct += (preds == y).sum().item()
            v_total += y.size(0)
            v_loss += loss.item()

    val_acc = v_correct / v_total * 100
    avg_val_loss = v_loss / len(val_loader)

    print(f"\n Epoch {epoch + 1} Results:")
    print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
    if DEVICE == "cuda":
        print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB / {torch.cuda.max_memory_allocated()/1024**2:.0f}MB peak")
        torch.cuda.empty_cache()  
    
    if val_acc > best_acc:
        best_acc = val_acc
        try:
            best_model_path = os.path.join(SCRIPT_DIR, "deepfake_model_v2_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"    Saved best model to: {best_model_path}")
        except Exception as e:
            print(f"    ERROR: Failed to save best model: {e}")


try:
    final_model_path = os.path.join(SCRIPT_DIR, "deepfake_model_v2.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n Final model saved to: {final_model_path}")
except Exception as e:
    print(f"\n ERROR: Failed to save final model: {e}")

print("\n" + "=" * 50)
print(" TRAINING COMPLETE ")
print("=" * 50)
print(f"   Device Used: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
print(f"   Best Validation Accuracy: {best_acc:.2f}%")
print(f"   Best Model: {os.path.join(SCRIPT_DIR, 'deepfake_model_v2_best.pth')}")
print(f"   Final Model: {os.path.join(SCRIPT_DIR, 'deepfake_model_v2.pth')}")
print("=" * 50)

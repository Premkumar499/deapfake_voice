
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import random
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2Model, Wav2Vec2Processor

from tqdm import tqdm



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

OLD_MODEL_PATH = os.path.join(SCRIPT_DIR, "deepfake_model_best.pth")

BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "deepfake_gpu_best.pth")

FINAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "deepfake_gpu_final.pth")


SAMPLE_RATE = 16000
MAX_SECONDS = 3
MAX_LEN = SAMPLE_RATE * MAX_SECONDS

BATCH_SIZE = 16
EPOCHS = 25

LR = 1e-5



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_AMP = DEVICE.type == "cuda"

print("\n" + "="*50)
print("DEVICE CONFIGURATION")
print("="*50)

if DEVICE.type == "cuda":

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:
    print("WARNING: GPU NOT FOUND, USING CPU")



def safe_load(path):

    try:

        audio, _ = librosa.load(path, sr=SAMPLE_RATE)

        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]

        else:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))

        return audio.astype(np.float32)

    except Exception:
        return np.zeros(MAX_LEN, dtype=np.float32)



def spec_augment(audio):

    try:

        spec = librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE)

        spec = librosa.power_to_db(spec)

        t = spec.shape[1]
        f = spec.shape[0]

        t0 = random.randint(0, max(1, t-10))
        spec[:, t0:t0+10] = 0

        f0 = random.randint(0, max(1, f-5))
        spec[f0:f0+5, :] = 0

        audio = librosa.feature.inverse.mel_to_audio(
            librosa.db_to_power(spec),
            sr=SAMPLE_RATE
        )

        return audio

    except:
        return audio



def mixup(audio):

    lam = np.random.beta(0.2,0.2)

    return lam*audio + (1-lam)*audio[::-1]



def augment(audio):

    try:

        if random.random()<0.3:
            audio += np.random.randn(len(audio))*0.002

        if random.random()<0.3:
            audio = librosa.effects.pitch_shift(
                audio,
                SAMPLE_RATE,
                random.randint(-2,2)
            )

        if random.random()<0.3:
            audio = spec_augment(audio)

        if random.random()<0.2:
            audio = mixup(audio)

        return audio

    except:
        return audio



def extract_features(audio):

    try:

        rms = librosa.feature.rms(audio)[0]

        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        flat = librosa.feature.spectral_flatness(audio)[0]

        mfcc = librosa.feature.mfcc(audio, sr=SAMPLE_RATE)

        pitch, _, _ = librosa.pyin(audio, fmin=50, fmax=300)

        pitch = pitch[~np.isnan(pitch)]

        silence = librosa.effects.split(audio)

        silence_ratio = 1 - (np.sum([e-s for s,e in silence]) / len(audio))

        features = np.array([
            np.mean(rms),
            np.var(rms),
            np.mean(zcr),
            np.mean(flat),
            np.var(audio),
            np.mean(np.var(mfcc,axis=1)),
            np.mean(pitch) if len(pitch)>0 else 0,
            np.var(pitch) if len(pitch)>0 else 0,
            silence_ratio,
            np.std(audio)
        ], dtype=np.float32)

        features = (features - np.mean(features)) / (np.std(features)+1e-6)

        return features

    except:

        return np.zeros(10,dtype=np.float32)



class AudioDataset(Dataset):

    def __init__(self, files, labels, train=True):

        self.files = files
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        audio = safe_load(self.files[idx])

        if self.train:
            audio = augment(audio)

        features = extract_features(audio)

        return audio, features, self.labels[idx]



def load_split(split):

    files=[]
    labels=[]

    try:

        for label,name in enumerate(["real","fake"]):

            folder = os.path.join(DATASET_PATH, split, name)

            if not os.path.exists(folder):
                continue

            for f in os.listdir(folder):

                path = os.path.join(folder,f)

                files.append(path)
                labels.append(label)

    except Exception:
        traceback.print_exc()

    return files,labels


train_files,train_labels=load_split("train")
val_files,val_labels=load_split("val")



processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def collate(batch):

    audio,feat,label = zip(*batch)

    inputs = processor(
        list(audio),
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt"
    )

    return inputs.input_values, torch.tensor(feat), torch.tensor(label)


# GPU optimized loader
train_loader = DataLoader(
    AudioDataset(train_files,train_labels,True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    AudioDataset(val_files,val_labels,False),
    batch_size=BATCH_SIZE,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True
)



class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.extra = nn.Linear(10,64)

        self.drop = nn.Dropout(0.3)

        self.cls = nn.Linear(768+64,1)

    def forward(self,x,feat):

        x = self.encoder(x).last_hidden_state.mean(1)

        feat = self.extra(feat)

        x = torch.cat([x,feat],1)

        x = self.drop(x)

        return self.cls(x)


model = Model().to(DEVICE)



if os.path.exists(OLD_MODEL_PATH):

    try:
        model.load_state_dict(
            torch.load(OLD_MODEL_PATH,map_location=DEVICE),
            strict=False
        )
        print("Loaded old model")
    except:
        print("Failed loading old model")



for p in model.encoder.parameters():
    p.requires_grad=False



fake=train_labels.count(1)
real=train_labels.count(0)

pos_weight=torch.tensor(real/max(fake,1)).to(DEVICE)

criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)


optimizer=optim.AdamW(
    filter(lambda p:p.requires_grad,model.parameters()),
    lr=LR
)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,EPOCHS
)

scaler=torch.cuda.amp.GradScaler(enabled=USE_AMP)


# ==========================================================
# TRAIN LOOP
# ==========================================================

best_acc=0

torch.cuda.empty_cache()

for epoch in range(EPOCHS):

    print("\nEpoch:",epoch+1)

    if epoch==3:

        for p in model.encoder.encoder.layers[-2:].parameters():
            p.requires_grad=True

    model.train()

    correct,total=0,0

    for x,feat,y in tqdm(train_loader):

        x=x.to(DEVICE,non_blocking=True)
        feat=feat.to(DEVICE,non_blocking=True)

        y=y.float().unsqueeze(1).to(DEVICE,non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):

            out=model(x,feat)

            loss=criterion(out,y)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        pred=(torch.sigmoid(out)>0.5).int()

        correct+=(pred==y.int()).sum().item()
        total+=y.size(0)

    train_acc=100*correct/total


    model.eval()

    correct,total=0,0

    with torch.no_grad():

        for x,feat,y in val_loader:

            x=x.to(DEVICE)
            feat=feat.to(DEVICE)

            out=model(x,feat)

            pred=(torch.sigmoid(out)>0.5).int().squeeze()

            correct+=(pred==y.to(DEVICE)).sum().item()

            total+=y.size(0)

    val_acc=100*correct/total

    print("Train:",train_acc,"Val:",val_acc)

    scheduler.step()

    if DEVICE.type=="cuda":

        print("GPU Memory:",
              round(torch.cuda.memory_allocated()/1024**3,2),
              "GB")

    if val_acc>best_acc:

        best_acc=val_acc

        torch.save(model.state_dict(),BEST_MODEL_PATH)


torch.save(model.state_dict(),FINAL_MODEL_PATH)

print("\nBEST ACC:",best_acc)
print("TRAINING COMPLETE")
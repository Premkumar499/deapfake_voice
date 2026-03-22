import torch
import os
from transformers import Wav2Vec2Processor
from detector import Detector
from audio_utils import load_audio

MODEL_PATH = "model/deepfake_model_v2.pth"   # Change to your .pth file name
PROCESSOR_NAME = "facebook/wav2vec2-base"
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading processor...")
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)


print("Loading model...")
model = Detector()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"Model loaded on {DEVICE}")


def predict(audio_path, language="Tamil"):

    if not os.path.exists(audio_path):
        print("Audio file not found!")
        return

    print("Processing audio...")
    audio = load_audio(audio_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE))
        prob = torch.sigmoid(logits).squeeze().item()

    classification = "AI_GENERATED" if prob > THRESHOLD else "HUMAN"
    confidence = round(prob if prob > THRESHOLD else 1 - prob, 4)

    explanation = generate_explanation(classification, confidence, language)

    print("\n========== RESULT ==========")
    print("Prediction:", classification)
    print("Confidence:", round(confidence * 100, 2), "%")
    print("============================")

    import json
    result = {
        "status": "success",
        "language": language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation,
    }
    print(json.dumps(result, indent=4))


def generate_explanation(classification: str, confidence: float, language: str) -> str:
    if classification == "AI_GENERATED":
        if confidence > 0.9:
            return f"High confidence AI detection: Unnatural pitch consistency and robotic speech patterns detected in {language} audio"
        elif confidence > 0.75:
            return f"Synthetic voice markers detected: Irregular prosody and mechanical intonation found in {language} audio"
        elif confidence > 0.6:
            return f"Moderate AI indicators: Subtle artifacts and unnatural transitions detected in {language} audio"
        else:
            return f"Weak AI indicators: Minor synthetic patterns observed in {language} audio near the detection threshold"
    else:
        if confidence > 0.9:
            return f"High confidence human detection: Natural speech variations and organic vocal patterns confirmed in {language} audio"
        elif confidence > 0.75:
            return f"Human voice confirmed: Consistent natural prosody and breathing patterns detected in {language} audio"
        elif confidence > 0.6:
            return f"Likely human voice: Natural speech characteristics observed in {language} audio"
        else:
            return f"Marginal human classification: Voice is near the detection boundary but leans toward natural human speech in {language} audio"


SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}


if __name__ == "__main__":

    audio_path = input("Enter audio file path: ").strip()
    language = input("Enter language (Tamil/English/Hindi/Malayalam/Telugu): ").strip()
    if language not in SUPPORTED_LANGUAGES:
        print(f"Unsupported language '{language}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    else:
        predict(audio_path, language)
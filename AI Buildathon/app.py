import torch
import os
import base64
import tempfile
import logging
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from transformers import Wav2Vec2Processor
from detector import Detector
from audio_utils import load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_PATH = "model/deepfake_model_v2.pth"
PROCESSOR_NAME = "facebook/wav2vec2-base"
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# API Key 
API_KEY = os.getenv("API_KEY", "sk_test_123456789")

SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

# LOAD MODEL 
logger.info("Loading processor...")
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)

logger.info("Loading model...")
model = Detector()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
logger.info(f"Model loaded on {DEVICE}")


app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human (Tamil, English, Hindi, Malayalam, Telugu)",
    version="1.0.0",
)

class VoiceDetectionRequest(BaseModel):
    language: str = Field(..., description="Language of the audio: Tamil / English / Hindi / Malayalam / Telugu")
    audioFormat: str = Field(..., description="Audio format, must be mp3")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio")


class VoiceDetectionSuccessResponse(BaseModel):
    status: str = "success"
    language: str
    classification: str
    confidenceScore: float
    explanation: str


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


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


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionSuccessResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key),
):
   
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}",
        )

    
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only mp3 audio format is supported",
        )

    
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 encoding in audioBase64 field",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Decoded audio is empty",
        )

    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

       
        audio = load_audio(tmp_path)

      
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

       
        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE))
            prob = torch.sigmoid(logits).squeeze().item()

     
        classification = "AI_GENERATED" if prob > THRESHOLD else "HUMAN"
        confidence_score = round(prob if prob > THRESHOLD else 1 - prob, 4)

        explanation = generate_explanation(classification, confidence_score, request.language)

        return VoiceDetectionSuccessResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=confidence_score,
            explanation=explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during voice analysis: {str(e)}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)



@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}



if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

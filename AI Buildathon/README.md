---
title: AI Voice Detection API
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AI Voice Detection API

Detects whether a voice sample is **AI-generated** or **Human** speech.

## Supported Languages
- Tamil
- English
- Hindi
- Malayalam
- Telugu

## API Endpoint

### `POST /api/voice-detection`

**Headers:**
| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `x-api-key` | Your API key |

**Request Body:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-mp3>"
}
```

**Response:**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "HUMAN",
  "confidenceScore": 0.9234,
  "explanation": "High confidence human detection: Natural speech variations..."
}
```

### `GET /health`
Health check endpoint.

## Tech Stack
- **Model:** Wav2Vec2-base (fine-tuned for deepfake detection)
- **Framework:** FastAPI
- **Runtime:** Docker on Hugging Face Spaces
# guvi_final
# guvi_hackathon

# Kokoro-TTS: A FastAPI-Based Text-to-Speech Service

Kokoro-TTS is a customizable Text-to-Speech (TTS) service built using FastAPI. It supports real-time audio streaming and file-based playback, allowing integration into various applications. The service is designed for scalability, running as a Dockerized application, and can be integrated with other services like Home Assistant.

---

## Features

- **Real-Time TTS Streaming**: Stream audio as it's generated for low-latency applications.
- **Multiple Formats**: Supports WAV, MP3, and PCM output formats.
- **Customizable Voices**: Integrates with voice packs to provide different accents and styles.
- **Dockerized**: Easy deployment and scaling using Docker Compose.

---

## Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**

If running locally, ensure the following Python dependencies are installed:

### Python Dependencies
Found in `requirements.txt`:
```plaintext
fastapi
uvicorn
pydantic
torch
soundfile
phonemizer
transformers
scipy
munch
nltk
```

### Running the Service Using Docker Compose

Use the provided docker-compose.yaml to build and run the service. Build and start the containers:

```bash
docker-compose up --build
```

--- 

## Access the service at:
Kokoro-TTS API: http://localhost:8880/v1/audio/speech

API Endpoints
POST /v1/audio/speech

Generate audio for given text input.
Request Body:

```json
{
  "model": "kokoro",
  "input": "Hello, this is a test message.",
  "voice": "af_sky",
  "response_format": "wav",
  "speed": 1.0,
  "stream": true
}
```

model: TTS model to use (default: kokoro).
input: Text to convert to speech.
voice: Voice pack identifier.
response_format: Output format (wav, mp3, or pcm).
speed: Speech speed multiplier.
stream: Whether to stream audio (true or false).

### Example cURL Command:
```json
curl -X POST http://localhost:8880/v1/audio/speech \
-H "Content-Type: application/json" \
-d '{
  "model": "kokoro",
  "input": "Hello, world!",
  "voice": "af_sky",
  "response_format": "wav",
  "speed": 1.0,
  "stream": true
}' --output test_output.wav
```

## Credits
This project uses the Kokoro-82M model from hexgrad, available on Hugging Face. The Kokoro-82M model powers the TTS capabilities of this service, providing high-quality and customizable voice generation.

## License
This project is licensed under MIT License.

## Acknowledgments
Built using FastAPI.
Inspired by modern TTS technologies and open-source tools.
Special thanks to the creators of Kokoro-82M for their contribution to open-source TTS technology.
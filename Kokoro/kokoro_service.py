from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
import torch
import soundfile as sf
from kokoro import generate
from models import build_model
import os
import uuid
import logging
from io import BytesIO
import re
import time

# Configuration
DEFAULT_VOICE = "af_sky"
SAMPLE_RATE = 24000
OUTPUT_DIR = "/app/tts_outputs"
PYTORCH_MODEL_PATH = "kokoro-v0_19.pth"
ONNX_MODEL_PATH = "kokoro-v0_19.onnx"
VOICE_DIR = "voices"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("KOKORO_API_KEY", "not-needed")
HOST = os.getenv("SERVICE_HOST", "localhost")

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Initialize model
logging.info(f"Running on device: {DEVICE}")
MODEL = build_model(PYTORCH_MODEL_PATH, DEVICE)

# FastAPI app
app = FastAPI()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = True  # Default to True

# Custom Sentence Splitter (Preserves Dialogues & Limits Length)
def split_text(input_text, max_length=150):
    """
    Custom sentence splitter that preserves dialogues and ensures sentence integrity.
    Limits chunk size to prevent TTS model errors.
    """
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", input_text)  # Ensure proper sentence spacing
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|")\s', text)

    # Ensure each chunk is within max_length
    result = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence
        else:
            result.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        result.append(current_chunk.strip())

    return result

@app.post("/v1/audio/speech")
async def synthesize_tts(
    request: TTSRequest,
    fastapi_request: Request,
    authorization: str = Header(None)
):
    logging.debug(f"Request received: {request.dict()}")
    logging.debug(f"Authorization Header: {authorization}")

    # Validate API key
    if API_KEY != "not-needed" and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Validate model
    if request.model != "kokoro":
        raise HTTPException(status_code=400, detail="Unsupported model")

    # Validate response format
    if request.response_format not in ["wav", "mp3", "pcm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response format: {request.response_format}"
        )

    # Validate voice
    voice_name = request.voice if request.voice else DEFAULT_VOICE
    voicepack_path = os.path.join(VOICE_DIR, f"{voice_name}.pt")
    logging.debug(f"Validating voice: {voice_name}, path: {voicepack_path}")

    if not os.path.exists(voicepack_path):
        logging.error(f"Voice pack not found: {voicepack_path}")
        voice_name = DEFAULT_VOICE
        voicepack_path = os.path.join(VOICE_DIR, f"{DEFAULT_VOICE}.pt")
        logging.debug(f"Using default voice: {voice_name}, path: {voicepack_path}")
        
        if not os.path.exists(voicepack_path):
            logging.error(f"Default voice pack not found: {voicepack_path}")
            raise HTTPException(status_code=500, detail="Default voice pack not found")

    # Load the voice pack
    try:
        voicepack = torch.load(voicepack_path, weights_only=True).to(DEVICE)
        logging.debug("Voice pack loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading voice pack: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading voice pack: {str(e)}")

    def audio_stream_generator():
        total_chunks = 0
        failed_chunks = []

        try:
            for chunk in split_text(request.input):
                if len(chunk.strip()) < 2:
                    continue

                attempt = 0
                max_attempts = 3
                snippet = None

                while attempt < max_attempts:
                    try:
                        snippet, _ = generate(MODEL, chunk, voicepack, lang=voice_name[0], speed=request.speed)
                        if snippet is not None:
                            break
                    except Exception as e:
                        logging.warning(f"⚠️ Attempt {attempt + 1} failed for chunk: {chunk}. Error: {e}")
                    
                    attempt += 1

                if snippet is None:
                    logging.warning(f"Skipping chunk after {max_attempts} failed attempts: {chunk}")
                    failed_chunks.append(chunk)
                    continue

                total_chunks += 1
                audio_bytes = BytesIO()
                sf.write(audio_bytes, snippet, SAMPLE_RATE, format=request.response_format.upper())
                audio_bytes.seek(0)
                yield audio_bytes.read()
                time.sleep(0.2)

        except Exception as e:
            logging.error(f"Critical error in audio_stream_generator: {e}")
        finally:
            logging.debug(f"Total chunks processed: {total_chunks}")
            if failed_chunks:
                logging.error(f"The following chunks were skipped: {failed_chunks}")
            yield b""

    if request.stream:
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={"Transfer-Encoding": "chunked", "Connection": "keep-alive"}
        )
    else:
        response_audio = b''.join(audio_stream_generator())
        if not response_audio:
            logging.error("No audio generated for the entire text. Check chunking or TTS model errors.")
            raise HTTPException(status_code=500, detail="TTS generation failed.")
        return Response(content=response_audio, media_type="audio/wav")

@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    file_path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)

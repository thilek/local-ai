from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import soundfile as sf
from kokoro import generate
from models import build_model
import os
import uuid
import logging
from io import BytesIO
from nltk.tokenize import sent_tokenize  # Added for improved text splitting
import time

# Configuration
DEFAULT_VOICE = "af_sky"
SAMPLE_RATE = 24000
OUTPUT_DIR = "/app/tts_outputs"
PYTORCH_MODEL_PATH = "kokoro-v0_19.pth"
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

@app.post("/v1/audio/speech")
async def synthesize_tts(
    request: TTSRequest,
    fastapi_request: Request,
    authorization: str = Header(None)
):
    logging.debug(f"Request received: {request.dict()}")
    logging.debug(f"Authorization Header: {authorization}")

    # Log full incoming request (headers and body)
    headers = dict(fastapi_request.headers)
    logging.debug(f"Request Headers: {headers}")
    try:
        body = await fastapi_request.body()
        logging.debug(f"Request Body: {body.decode('utf-8')}")
    except Exception as e:
        logging.error(f"Error reading request body: {e}")

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

    # Helper function for better text splitting
    def split_text(input_text):
        """Split input text into meaningful chunks."""
        return sent_tokenize(input_text)
    
    # Generate TTS audio
    def audio_stream_generator():
        """Generator function to yield audio chunks."""
        total_chunks = 0
        try:
            for chunk in split_text(request.input):
                if len(chunk.strip()) < 2:
                    logging.debug(f"Skipping empty or short chunk: {chunk}")
                    continue

                logging.debug(f"Processing chunk: {chunk}")
                total_chunks += 1

                try:
                    # Generate audio for the chunk
                    snippet, _ = generate(MODEL, chunk, voicepack, lang=voice_name[0], speed=request.speed)

                    if snippet is not None:
                        # Convert the snippet to a list if needed
                        snippet = snippet.tolist() if hasattr(snippet, "tolist") else snippet

                        # Write audio chunk to buffer and yield it
                        audio_bytes = BytesIO()
                        sf.write(audio_bytes, snippet, SAMPLE_RATE, format=request.response_format.upper())
                        audio_bytes.seek(0)
                        chunk_data = audio_bytes.read()
                        logging.debug(f"Generated chunk of size: {len(chunk_data)} bytes")
                        yield chunk_data
                        time.sleep(0.1)  # Add a small delay to prevent skipping
                    else:
                        logging.warning(f"No audio generated for chunk: {chunk}")

                except Exception as e:
                    logging.warning(f"Failed to process chunk: {chunk}. Error: {e}")

        except Exception as e:
            logging.error(f"Error in audio_stream_generator: {e}")
            yield b""
        finally:
            logging.debug(f"Total chunks processed: {total_chunks}")

    if request.stream:
        # Streaming response
        try:
            logging.debug("Streaming audio...")
            return StreamingResponse(
                audio_stream_generator(),
                media_type="audio/wav",
                headers={"Transfer-Encoding": "chunked"}
            )
        except Exception as e:
            logging.error(f"Error streaming audio: {e}")
            raise HTTPException(status_code=500, detail=f"Error streaming audio: {str(e)}")
    else:
        # Save audio to a file
        output_file = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.{request.response_format}")
        logging.debug(f"Saving audio to {output_file}")
        try:
            audio = []
            for chunk in split_text(request.input):
                if len(chunk.strip()) < 2:
                    continue  # Skip empty or invalid chunks

                snippet, _ = generate(MODEL, chunk, voicepack, lang=voice_name[0], speed=request.speed)
                if snippet is not None:
                    snippet = snippet.tolist() if hasattr(snippet, "tolist") else snippet
                    audio.extend(snippet)

            if not audio:
                raise ValueError("No audio data generated.")

            sf.write(output_file, audio, SAMPLE_RATE, format=request.response_format.upper())
            logging.debug(f"Audio saved successfully at {output_file}")
        except Exception as e:
            logging.error(f"Error saving audio: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving audio: {str(e)}")

        return {
            "audio_url": f"http://{HOST}:8880/audio/{os.path.basename(output_file)}"
        }

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

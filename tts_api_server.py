"""
Kokoro TTS API Server
--------------------
A Flask-based API server that provides streaming TTS audio generation.
Supports multiple voices and real-time audio streaming.

Key Features:
- RESTful API endpoint for TTS generation
- Streaming audio response
- Multiple voice support
- Configurable speed and format
- Cross-platform compatibility

Dependencies:
- flask: Web framework
- kokoro: Official Kokoro TTS library
- soundfile: Audio file handling
- pydub: Audio format conversion
"""

from flask import Flask, request, Response, jsonify
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
from typing import Union, List, Optional, Tuple, Dict, Any
import io
import json
import logging
from models import (
    list_available_voices, build_model,
    generate_speech, download_voice_files
)
from kokoro import KPipeline
import speed_dial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define path type for consistent handling
PathLike = Union[str, Path]

# Configuration validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        logger.warning(f"Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

# Global configuration
CONFIG_FILE = Path("tts_config.json")  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = Path("outputs")    # Directory for generated audio files
SAMPLE_RATE = validate_sample_rate(24000)  # Validated sample rate

# Initialize model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None

LANG_MAP = {
    "af_": "a", "am_": "a",
    "bf_": "b", "bm_": "b",
    "jf_": "j", "jm_": "j",
    "zf_": "z", "zm_": "z",
    "ef_": "e", "em_": "e",
    "ff_": "f",
    "hf_": "h", "hm_": "h",
    "if_": "i", "im_": "i",
    "pf_": "p", "pm_": "p",
}
pipelines = {}

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        global model
        if model is None:
            logger.info("Initializing model and downloading voices...")
            model = build_model(None, device)

        voices = list_available_voices()
        if not voices:
            logger.info("No voices found after initialization. Attempting to download...")
            download_voice_files()  # Try downloading again
            voices = list_available_voices()

        logger.info(f"Available voices: {voices}")
        return voices
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        return []

def get_pipeline_for_voice(voice_name: str) -> KPipeline:
    """
    Determine the language code from the voice prefix and return the associated pipeline.
    """
    prefix = voice_name[:3].lower()
    lang_code = LANG_MAP.get(prefix, "a")
    if lang_code not in pipelines:
        logger.info(f"Creating pipeline for lang_code='{lang_code}'")
        pipelines[lang_code] = KPipeline(lang_code=lang_code, model=True)
    return pipelines[lang_code]

def generate_tts_stream(voice_name: str, text: str, speed: float = 1.0):
    """Generate TTS audio as a streaming generator.

    Args:
        voice_name: Name of the voice to use
        text: Text to convert to speech
        speed: Speech speed modifier

    Yields:
        Audio chunks as bytes
    """
    global model

    try:
        # Initialize model if needed
        if model is None:
            logger.info("Initializing model...")
            model = build_model(None, device)

        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Limit extremely long texts to prevent memory issues
        MAX_CHARS = 5000
        if len(text) > MAX_CHARS:
            logger.warning(f"Text exceeds {MAX_CHARS} characters. Truncating to prevent memory issues.")
            text = text[:MAX_CHARS] + "..."

        # Validate voice path using Path for consistent handling
        voice_path = Path("voices").absolute() / f"{voice_name}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        logger.info(f"Generating speech for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        logger.info(f"Using voice: {voice_name}")

        try:
            if voice_name.startswith(tuple(LANG_MAP.keys())):
                pipeline = get_pipeline_for_voice(voice_name)
                generator = pipeline(text, voice=voice_path, speed=speed, split_pattern=r'\n+')
            else:
                generator = model(text, voice=voice_path, speed=speed, split_pattern=r'\n+')

            # Stream audio chunks
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    
                    # Convert to bytes for streaming
                    audio_bytes = audio.numpy().tobytes()
                    yield audio_bytes
                    
                    logger.debug(f"Generated segment: {gs}")
                    if ps:  # Only log phonemes if available
                        logger.debug(f"Phonemes: {ps}")

        except Exception as e:
            raise Exception(f"Error in speech generation: {e}")

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        yield b''  # Return empty bytes on error

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Get available voices on startup
    voices = get_available_voices()
    if not voices:
        logger.warning("No voices found! Please check the voices directory.")

    @app.route('/tts/', methods=['POST'])
    def tts_endpoint():
        """TTS endpoint that accepts text and returns streaming audio."""
        try:
            # Parse request data
            if request.is_json:
                data = request.get_json()
                text = data.get('text', '')
                voice_name = data.get('voice', voices[0] if voices else None)
                speed = float(data.get('speed', 1.0))
                format_type = data.get('format', 'wav').lower()
            else:
                # Form data fallback
                text = request.form.get('text', '')
                voice_name = request.form.get('voice', voices[0] if voices else None)
                speed = float(request.form.get('speed', 1.0))
                format_type = request.form.get('format', 'wav').lower()

            # Validate inputs
            if not text:
                return jsonify({'error': 'Text parameter is required'}), 400
            
            if not voice_name:
                return jsonify({'error': 'Voice parameter is required'}), 400
            
            if voice_name not in voices:
                return jsonify({'error': f'Voice "{voice_name}" not found. Available voices: {voices}'}), 400
            
            if speed < 0.1 or speed > 3.0:
                return jsonify({'error': 'Speed must be between 0.1 and 3.0'}), 400

            # Set response headers for streaming
            headers = {
                'Content-Type': 'audio/wav',
                'Cache-Control': 'no-cache',
                'X-Voice': voice_name,
                'X-Speed': str(speed),
                'X-Text-Length': str(len(text))
            }

            # Create streaming response
            def generate():
                try:
                    # Generate audio stream
                    audio_chunks = []
                    for chunk in generate_tts_stream(voice_name, text, speed):
                        if chunk:
                            audio_chunks.append(chunk)
                    
                    if not audio_chunks:
                        logger.error("No audio generated")
                        return
                    
                    # Combine all chunks
                    if len(audio_chunks) == 1:
                        final_audio = audio_chunks[0]
                    else:
                        # Convert bytes back to tensor for concatenation
                        audio_tensors = [torch.frombuffer(chunk, dtype=torch.float32) for chunk in audio_chunks]
                        final_audio = torch.cat(audio_tensors, dim=0)
                    
                    # Convert to WAV format
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, final_audio.numpy(), SAMPLE_RATE, format='WAV')
                    audio_buffer.seek(0)
                    
                    # Stream the audio data
                    yield audio_buffer.read()
                    
                except Exception as e:
                    logger.error(f"Error in audio generation: {e}")
                    yield b''

            return Response(generate(), headers=headers)

        except Exception as e:
            logger.error(f"Error in TTS endpoint: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/tts/voices', methods=['GET'])
    def get_voices():
        """Get list of available voices."""
        try:
            voices = get_available_voices()
            return jsonify({
                'voices': voices,
                'count': len(voices)
            })
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/tts/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        try:
            voices = get_available_voices()
            return jsonify({
                'status': 'healthy',
                'voices_available': len(voices),
                'device': device,
                'model_loaded': model is not None
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

    @app.route('/tts/', methods=['GET'])
    def tts_info():
        """Get TTS service information."""
        return jsonify({
            'service': 'Kokoro TTS API',
            'version': '1.0.0',
            'endpoints': {
                'POST /tts/': 'Generate streaming TTS audio',
                'GET /tts/voices': 'Get available voices',
                'GET /tts/health': 'Health check'
            },
            'parameters': {
                'text': 'Text to convert to speech (required)',
                'voice': 'Voice name (required)',
                'speed': 'Speech speed (0.1-3.0, default: 1.0)',
                'format': 'Output format (wav, mp3, aac, default: wav)'
            }
        })

    return app

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    global model

    try:
        logger.info("Cleaning up resources...")

        # Clean up model resources
        if model is not None:
            logger.info("Releasing model resources...")

            # Clear voice dictionary to release memory
            if hasattr(model, 'voices') and model.voices is not None:
                try:
                    voice_count = len(model.voices)
                    for voice_name in list(model.voices.keys()):
                        try:
                            # Release each voice explicitly
                            model.voices[voice_name] = None
                        except:
                            pass
                    model.voices.clear()
                    logger.info(f"Cleared {voice_count} voice references")
                except Exception as ve:
                    logger.error(f"Error clearing voices: {type(ve).__name__}: {ve}")

            # Clear model attributes that might hold tensors
            for attr_name in dir(model):
                if not attr_name.startswith('__') and hasattr(model, attr_name):
                    try:
                        attr = getattr(model, attr_name)
                        # Handle specific tensor attributes
                        if isinstance(attr, torch.Tensor):
                            if attr.is_cuda:
                                logger.info(f"Releasing CUDA tensor: {attr_name}")
                                setattr(model, attr_name, None)
                        elif hasattr(attr, 'to'):  # Module or Tensor-like object
                            setattr(model, attr_name, None)
                    except:
                        pass

            # Delete model reference
            try:
                del model
                model = None
                logger.info("Model reference deleted")
            except Exception as me:
                logger.error(f"Error deleting model: {type(me).__name__}: {me}")

        # Clear CUDA memory explicitly
        if torch.cuda.is_available():
            try:
                # Get initial memory usage
                try:
                    initial = torch.cuda.memory_allocated()
                    initial_mb = initial / (1024 * 1024)
                    logger.info(f"CUDA memory before cleanup: {initial_mb:.2f} MB")
                except:
                    pass

                # Free memory
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()

                # Force synchronization
                try:
                    torch.cuda.synchronize()
                except:
                    pass

                # Get final memory usage
                try:
                    final = torch.cuda.memory_allocated()
                    final_mb = final / (1024 * 1024)
                    freed_mb = (initial - final) / (1024 * 1024)
                    logger.info(f"CUDA memory after cleanup: {final_mb:.2f} MB (freed {freed_mb:.2f} MB)")
                except:
                    pass
            except Exception as ce:
                logger.error(f"Error clearing CUDA memory: {type(ce).__name__}: {ce}")

        # Restore original functions
        try:
            from models import _cleanup_monkey_patches
            _cleanup_monkey_patches()
            logger.info("Monkey patches restored")
        except Exception as pe:
            logger.error(f"Error restoring monkey patches: {type(pe).__name__}: {pe}")

        # Final garbage collection
        try:
            import gc
            collected = gc.collect()
            logger.info(f"Garbage collection completed: {collected} objects collected")
        except Exception as gce:
            logger.error(f"Error during garbage collection: {type(gce).__name__}: {gce}")

        logger.info("Cleanup completed")

    except Exception as e:
        logger.error(f"Error during cleanup: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Register cleanup for normal exit
import atexit
atexit.register(cleanup_resources)

# Register cleanup for signals
import signal
import sys

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register for common signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, signal_handler)
    except (ValueError, AttributeError):
        # Some signals might not be available on all platforms
        pass

if __name__ == "__main__":
    app = create_app()
    
    # Configuration
    host = os.environ.get('TTS_HOST', '127.0.0.1')
    port = int(os.environ.get('TTS_PORT', 5000))
    debug = os.environ.get('TTS_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting TTS API server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    finally:
        # Ensure cleanup even if Flask encounters an error
        cleanup_resources() 
"""Models module for Kokoro TTS Local with local loading support"""
from typing import Optional, Tuple, List
import torch
from kokoro import KPipeline
import os
import json
import codecs
from pathlib import Path
import numpy as np
import shutil
import threading
from tts_config import TTSConfig

# Set environment variables for proper encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Setup for safer monkey-patching
import atexit
import signal
import sys
import logging

logger = logging.getLogger(__name__)

# Track whether patches have been applied
_patches_applied = {
    'json_load': False,
    'load_voice': False
}

def _cleanup_monkey_patches():
    """Restore original functions that were monkey-patched"""
    try:
        if _patches_applied['json_load'] and _original_json_load is not None:
            restore_json_load()
            _patches_applied['json_load'] = False
            logger.info("Restored original json.load function")
    except Exception as e:
        logger.warning(f"Error restoring json.load: {e}")

    try:
        if _patches_applied['load_voice']:
            restore_original_load_voice()
            _patches_applied['load_voice'] = False
            logger.info("Restored original KPipeline.load_voice function")
    except Exception as e:
        logger.warning(f"Error restoring KPipeline.load_voice: {e}")

# Register cleanup for normal exit
atexit.register(_cleanup_monkey_patches)

# Register cleanup for signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, lambda signum, frame: (
            logger.info(f"Received signal {signum}, cleaning up..."),
            _cleanup_monkey_patches(),
            sys.exit(1)
        ))
    except (ValueError, AttributeError):
        # Some signals might not be available on all platforms
        pass

# Global pipeline instance with thread safety
_pipeline = None
_pipeline_lock = threading.Lock()

# List of available voice files (54 voices across 8 languages)
VOICE_FILES = [
    # American English Female voices (11 voices)
    "af_heart.pt", "af_alloy.pt", "af_aoede.pt", "af_bella.pt", "af_jessica.pt",
    "af_kore.pt", "af_nicole.pt", "af_nova.pt", "af_river.pt", "af_sarah.pt", "af_sky.pt",

    # American English Male voices (9 voices)
    "am_adam.pt", "am_echo.pt", "am_eric.pt", "am_fenrir.pt", "am_liam.pt",
    "am_michael.pt", "am_onyx.pt", "am_puck.pt", "am_santa.pt",

    # British English Female voices (4 voices)
    "bf_alice.pt", "bf_emma.pt", "bf_isabella.pt", "bf_lily.pt",

    # British English Male voices (4 voices)
    "bm_daniel.pt", "bm_fable.pt", "bm_george.pt", "bm_lewis.pt",

    # Japanese voices (5 voices)
    "jf_alpha.pt", "jf_gongitsune.pt", "jf_nezumi.pt", "jf_tebukuro.pt", "jm_kumo.pt",

    # Mandarin Chinese voices (8 voices)
    "zf_xiaobei.pt", "zf_xiaoni.pt", "zf_xiaoxiao.pt", "zf_xiaoyi.pt",
    "zm_yunjian.pt", "zm_yunxi.pt", "zm_yunxia.pt", "zm_yunyang.pt",

    # Spanish voices (3 voices)
    "ef_dora.pt", "em_alex.pt", "em_santa.pt",

    # French voices (1 voice)
    "ff_siwis.pt",

    # Hindi voices (4 voices)
    "hf_alpha.pt", "hf_beta.pt", "hm_omega.pt", "hm_psi.pt",

    # Italian voices (2 voices)
    "if_sara.pt", "im_nicola.pt",

    # Brazilian Portuguese voices (3 voices)
    "pf_dora.pt", "pm_alex.pt", "pm_santa.pt"
]

# Language code mapping for different languages
LANGUAGE_CODES = {
    'a': 'American English',
    'b': 'British English',
    'j': 'Japanese',
    'z': 'Mandarin Chinese',
    'e': 'Spanish',
    'f': 'French',
    'h': 'Hindi',
    'i': 'Italian',
    'p': 'Brazilian Portuguese'
}

# Patch KPipeline's load_voice method to use weights_only=False
original_load_voice = KPipeline.load_voice

def patched_load_voice(self, voice_path):
    """Load voice model with weights_only=False for compatibility"""
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    voice_name = Path(voice_path).stem
    try:
        voice_model = torch.load(voice_path, weights_only=False)
        if voice_model is None:
            raise ValueError(f"Failed to load voice model from {voice_path}")
        # Ensure device is set
        if not hasattr(self, 'device'):
            self.device = 'cpu'
        # Move model to device and store in voices dictionary
        self.voices[voice_name] = voice_model.to(self.device)
        return self.voices[voice_name]
    except Exception as e:
        logger.error(f"Error loading voice {voice_name}: {e}")
        raise

# Apply the patch
KPipeline.load_voice = patched_load_voice
_patches_applied['load_voice'] = True

# Store original function for restoration if needed
def restore_original_load_voice():
    global _patches_applied
    if _patches_applied['load_voice']:
        KPipeline.load_voice = original_load_voice
        _patches_applied['load_voice'] = False

def patch_json_load():
    """Patch json.load to handle UTF-8 encoded files with special characters"""
    global _patches_applied, _original_json_load
    original_load = json.load
    _original_json_load = original_load  # Store for restoration

    def custom_load(fp, *args, **kwargs):
        try:
            # Try reading with UTF-8 encoding
            if hasattr(fp, 'buffer'):
                content = fp.buffer.read().decode('utf-8')
            else:
                content = fp.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise
        except UnicodeDecodeError:
            # If UTF-8 fails, try with utf-8-sig for files with BOM
            fp.seek(0)
            content = fp.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8-sig', errors='replace')
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise

    json.load = custom_load
    _patches_applied['json_load'] = True
    return original_load  # Return original for restoration

# Store the original load function for potential restoration
_original_json_load = None

def restore_json_load():
    """Restore the original json.load function"""
    global _original_json_load, _patches_applied
    if _original_json_load is not None and _patches_applied['json_load']:
        json.load = _original_json_load
        _original_json_load = None
        _patches_applied['json_load'] = False

def load_config(config_path: str) -> dict:
    """Load configuration file with proper encoding handling"""
    try:
        with codecs.open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to utf-8-sig if regular utf-8 fails
        with codecs.open(config_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

def download_voice_files(voice_files=None, repo_version="main", required_count=1):
    """Download voice files from Hugging Face Hub with local loading support"""
    config = TTSConfig()
    
    # Check if local voices are enabled
    if config.is_local_voices_enabled():
        local_voices_path = config.get_local_voices_path()
        if local_voices_path and local_voices_path.exists():
            logger.info(f"Using local voices from: {local_voices_path}")
            voice_files = list(local_voices_path.glob("*.pt"))
            if voice_files:
                logger.info(f"Found {len(voice_files)} local voice files")
                return [f.name for f in voice_files]
            else:
                logger.warning(f"No voice files found in local path: {local_voices_path}")
    
    # Fall back to downloading if local voices not available
    logger.info("Downloading voice files from Hugging Face Hub...")
    
    if voice_files is None:
        voice_files = VOICE_FILES

    # Create voices directory if it doesn't exist
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)

    downloaded_files = []
    repo_id = config.get_voices_config().get("repo_id", "hexgrad/Kokoro-82M")

    try:
        from huggingface_hub import hf_hub_download
        
        for voice_file in voice_files:
            voice_path = voices_dir / voice_file
            if voice_path.exists():
                logger.debug(f"Voice file already exists: {voice_file}")
                downloaded_files.append(voice_file)
                continue

            try:
                logger.info(f"Downloading voice: {voice_file}")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"voices/{voice_file}",
                    local_dir="voices",
                    force_download=False,
                    revision=repo_version
                )
                downloaded_files.append(voice_file)
                logger.info(f"Successfully downloaded: {voice_file}")
            except Exception as e:
                logger.error(f"Failed to download {voice_file}: {e}")

        if len(downloaded_files) < required_count:
            raise ValueError(f"Only {len(downloaded_files)} voice files downloaded, but {required_count} required")

        logger.info(f"Successfully downloaded {len(downloaded_files)} voice files")
        return downloaded_files

    except ImportError:
        logger.error("huggingface_hub not available. Please install it with: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Error downloading voice files: {e}")
        raise

def build_model(model_path: str, device: str, repo_version: str = "main") -> KPipeline:
    """Build and return the Kokoro pipeline with local loading support"""
    global _pipeline, _pipeline_lock
    config = TTSConfig()

    # Use a lock for thread safety
    with _pipeline_lock:
        # Double-check pattern to avoid race conditions
        if _pipeline is not None:
            return _pipeline

        try:
            # Patch json loading before initializing pipeline
            patch_json_load()

            # Handle model loading (local or download)
            if config.is_local_model_enabled():
                local_model_path = config.get_local_model_path()
                if local_model_path and local_model_path.exists():
                    model_path = str(local_model_path)
                    logger.info(f"Using local model: {model_path}")
                else:
                    logger.warning(f"Local model path not found: {local_model_path}")
                    logger.info("Falling back to download")
                    model_path = None
            else:
                model_path = None

            # Download model if needed
            if model_path is None:
                model_path = 'kokoro-v1_0.pth'

            model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                logger.info(f"Downloading model file {model_path}...")
                try:
                    from huggingface_hub import hf_hub_download
                    repo_id = config.get_model_config().get("repo_id", "hexgrad/Kokoro-82M")
                    model_filename = config.get_model_config().get("model_filename", "kokoro-v1_0.pth")
                    
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_filename,
                        local_dir=".",
                        force_download=True,
                        revision=repo_version
                    )
                    logger.info(f"Model downloaded to {model_path}")
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
                    raise ValueError(f"Could not download model: {e}") from e

            # Download config if it doesn't exist
            config_path = os.path.abspath("config.json")
            if not os.path.exists(config_path):
                logger.info("Downloading config file...")
                try:
                    from huggingface_hub import hf_hub_download
                    repo_id = config.get_model_config().get("repo_id", "hexgrad/Kokoro-82M")
                    config_filename = config.get_model_config().get("config_filename", "config.json")
                    
                    config_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=config_filename,
                        local_dir=".",
                        force_download=True,
                        revision=repo_version
                    )
                    logger.info(f"Config downloaded to {config_path}")
                except Exception as e:
                    logger.error(f"Error downloading config: {e}")
                    raise ValueError(f"Could not download config: {e}") from e

            # Download voice files - require at least one voice
            try:
                downloaded_voices = download_voice_files(repo_version=repo_version, required_count=1)
            except ValueError as e:
                logger.error(f"Error: Voice files download failed: {e}")
                raise ValueError("Voice files download failed") from e

            # Validate language code
            lang_code = 'a'  # Default to 'a' for American English
            supported_codes = list(LANGUAGE_CODES.keys())
            if lang_code not in supported_codes:
                logger.warning(f"Unsupported language code '{lang_code}'. Using 'a' (American English).")
                logger.info(f"Supported language codes: {', '.join(supported_codes)}")
                lang_code = 'a'

            # Initialize pipeline with validated language code
            pipeline_instance = KPipeline(lang_code=lang_code)
            if pipeline_instance is None:
                raise ValueError("Failed to initialize KPipeline - pipeline is None")

            # Store device parameter for reference in other operations
            pipeline_instance.device = device

            # Initialize voices dictionary if it doesn't exist
            if not hasattr(pipeline_instance, 'voices'):
                pipeline_instance.voices = {}

            # Try to load the first available voice with improved error handling
            voice_loaded = False
            for voice_file in downloaded_voices:
                # Use local voices path if configured
                if config.is_local_voices_enabled():
                    local_voices_path = config.get_local_voices_path()
                    voice_path = local_voices_path / voice_file
                else:
                    voice_path = os.path.abspath(os.path.join("voices", voice_file))
                
                if os.path.exists(voice_path):
                    try:
                        pipeline_instance.load_voice(str(voice_path))
                        logger.info(f"Successfully loaded voice: {voice_file}")
                        voice_loaded = True
                        break  # Successfully loaded a voice
                    except Exception as e:
                        logger.warning(f"Failed to load voice {voice_file}: {e}")
                        continue

            if not voice_loaded:
                logger.warning("Could not load any voice models")

            # Set the global _pipeline only after successful initialization
            _pipeline = pipeline_instance

        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            # Restore original json.load on error
            restore_json_load()
            raise

        return _pipeline

def list_available_voices() -> List[str]:
    """List all available voice models with local loading support"""
    config = TTSConfig()
    
    # Check if local voices are enabled
    if config.is_local_voices_enabled():
        local_voices_path = config.get_local_voices_path()
        if local_voices_path and local_voices_path.exists():
            voice_files = list(local_voices_path.glob("*.pt"))
            if voice_files:
                logger.info(f"Found {len(voice_files)} local voice files in {local_voices_path}")
                return [f.stem for f in sorted(voice_files, key=lambda f: f.stem.lower())]
    
    # Fall back to standard voices directory
    voices_dir = Path(os.path.abspath("voices"))

    # Create voices directory if it doesn't exist
    if not voices_dir.exists():
        logger.info(f"Creating voices directory at {voices_dir}")
        voices_dir.mkdir(exist_ok=True)
        return []

    # Get all .pt files in the voices directory
    voice_files = list(voices_dir.glob("*.pt"))

    # If we found voice files, return them
    if voice_files:
        return [f.stem for f in sorted(voice_files, key=lambda f: f.stem.lower())]

    # If no voice files in standard location, check if we need to do a one-time migration
    # This is legacy support for older installations
    alt_voices_path = Path(".") / "voices"
    if alt_voices_path.exists() and alt_voices_path.is_dir() and alt_voices_path != voices_dir:
        logger.info(f"Checking alternative voice location: {alt_voices_path.absolute()}")
        alt_voice_files = list(alt_voices_path.glob("*.pt"))

        if alt_voice_files:
            logger.info(f"Found {len(alt_voice_files)} voice files in alternate location")
            logger.info("Moving files to the standard voices directory...")

            # Process files in a batch for efficiency
            files_moved = 0
            for voice_file in alt_voice_files:
                target_path = voices_dir / voice_file.name
                if not target_path.exists():
                    try:
                        # Use copy2 to preserve metadata, then remove original if successful
                        shutil.copy2(str(voice_file), str(target_path))
                        files_moved += 1
                    except (OSError, IOError) as e:
                        logger.error(f"Error copying {voice_file.name}: {e}")

            if files_moved > 0:
                logger.info(f"Successfully moved {files_moved} voice files")
                return [f.stem for f in sorted(voices_dir.glob("*.pt"), key=lambda f: f.stem.lower())]

    logger.warning("No voice files found. Please run the application again to download voices.")
    return []

def get_language_code_from_voice(voice_name: str) -> str:
    """Get the appropriate language code from a voice name"""
    # Extract prefix from voice name
    prefix = voice_name[:2] if len(voice_name) >= 2 else 'af'

    # Map voice prefixes to language codes
    prefix_to_lang = {
        'af': 'a', 'am': 'a',  # American English
        'bf': 'b', 'bm': 'b',  # British English
        'jf': 'j', 'jm': 'j',  # Japanese
        'zf': 'z', 'zm': 'z',  # Mandarin Chinese
        'ef': 'e', 'em': 'e',  # Spanish
        'ff': 'f', 'fm': 'f',  # French
        'hf': 'h', 'hm': 'h',  # Hindi
        'if': 'i', 'im': 'i',  # Italian
        'pf': 'p', 'pm': 'p',  # Brazilian Portuguese
    }

    return prefix_to_lang.get(prefix, 'a')  # Default to American English

def load_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load a voice model in a thread-safe manner with local loading support"""
    config = TTSConfig()
    
    # Check if local voices are enabled
    if config.is_local_voices_enabled():
        local_voices_path = config.get_local_voices_path()
        if local_voices_path and local_voices_path.exists():
            voice_path = local_voices_path / f"{voice_name}.pt"
            if voice_path.exists():
                try:
                    voice_model = torch.load(str(voice_path), weights_only=False)
                    return voice_model.to(device)
                except Exception as e:
                    logger.error(f"Error loading local voice {voice_name}: {e}")
    
    # Fall back to standard loading
    voice_path = os.path.abspath(os.path.join("voices", f"{voice_name}.pt"))
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    
    try:
        voice_model = torch.load(voice_path, weights_only=False)
        return voice_model.to(device)
    except Exception as e:
        logger.error(f"Error loading voice {voice_name}: {e}")
        raise

def generate_speech(
    model: KPipeline,
    text: str,
    voice: str,
    lang: str = 'a',
    device: str = 'cpu',
    speed: float = 1.0
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Generate speech using the provided model and voice"""
    try:
        # Get language code from voice name if not provided
        if lang == 'a':
            lang = get_language_code_from_voice(voice)

        # Load voice
        voice_tensor = load_voice(voice, device)

        # Generate speech
        result = model(text, voice=voice_tensor, speed=speed)
        
        # Extract audio from result
        audio = None
        phonemes = None
        
        for graphemes, phonemes_text, audio_chunk in result:
            if audio_chunk is not None:
                if audio is None:
                    audio = audio_chunk
                else:
                    # Concatenate audio chunks
                    if isinstance(audio, torch.Tensor) and isinstance(audio_chunk, torch.Tensor):
                        audio = torch.cat([audio, audio_chunk], dim=0)
                    elif isinstance(audio, np.ndarray) and isinstance(audio_chunk, np.ndarray):
                        audio = np.concatenate([audio, audio_chunk])
                phonemes = phonemes_text

        return audio, phonemes

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None, None 
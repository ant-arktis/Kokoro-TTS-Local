"""
TTS Configuration Module
------------------------
Configuration management for Kokoro TTS Local with support for local model/voice loading.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TTSConfig:
    """Configuration manager for TTS settings"""
    
    def __init__(self, config_file: str = "tts_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                logger.info("Using default configuration")
        
        # Default configuration
        default_config = {
            "model": {
                "use_local": False,
                "local_path": None,
                "repo_id": "hexgrad/Kokoro-82M",
                "model_filename": "kokoro-v1_0.pth",
                "config_filename": "config.json"
            },
            "voices": {
                "use_local": False,
                "local_path": "voices",
                "repo_id": "hexgrad/Kokoro-82M",
                "required_count": 1
            },
            "server": {
                "host": "127.0.0.1",
                "port": 5000,
                "debug": False
            },
            "audio": {
                "sample_rate": 24000,
                "default_format": "wav"
            },
            "paths": {
                "output_dir": "outputs",
                "voices_dir": "voices",
                "models_dir": "models"
            }
        }
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self._save_config(self.config)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get("model", {})
    
    def get_voices_config(self) -> Dict[str, Any]:
        """Get voices configuration"""
        return self.config.get("voices", {})
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.config.get("server", {})
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration"""
        return self.config.get("audio", {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.config.get("paths", {})
    
    def is_local_model_enabled(self) -> bool:
        """Check if local model loading is enabled"""
        return self.config.get("model", {}).get("use_local", False)
    
    def is_local_voices_enabled(self) -> bool:
        """Check if local voices loading is enabled"""
        return self.config.get("voices", {}).get("use_local", False)
    
    def get_local_model_path(self) -> Optional[Path]:
        """Get local model path if configured"""
        if not self.is_local_model_enabled():
            return None
        
        path = self.config.get("model", {}).get("local_path")
        if path:
            return Path(path).resolve()
        return None
    
    def get_local_voices_path(self) -> Optional[Path]:
        """Get local voices path if configured"""
        if not self.is_local_voices_enabled():
            return None
        
        path = self.config.get("voices", {}).get("local_path")
        if path:
            return Path(path).resolve()
        return None
    
    def validate_local_paths(self) -> Dict[str, bool]:
        """Validate that configured local paths exist"""
        validation = {
            "model": False,
            "voices": False
        }
        
        # Check model path
        if self.is_local_model_enabled():
            model_path = self.get_local_model_path()
            if model_path and model_path.exists():
                validation["model"] = True
            else:
                logger.warning(f"Local model path does not exist: {model_path}")
        
        # Check voices path
        if self.is_local_voices_enabled():
            voices_path = self.get_local_voices_path()
            if voices_path and voices_path.exists():
                # Check if there are any .pt files
                voice_files = list(voices_path.glob("*.pt"))
                if voice_files:
                    validation["voices"] = True
                    logger.info(f"Found {len(voice_files)} voice files in {voices_path}")
                else:
                    logger.warning(f"No voice files found in {voices_path}")
            else:
                logger.warning(f"Local voices path does not exist: {voices_path}")
        
        return validation

def create_local_config_example():
    """Create an example configuration file for local loading"""
    example_config = {
        "model": {
            "use_local": True,
            "local_path": "./models/kokoro-v1_0.pth",
            "repo_id": "hexgrad/Kokoro-82M",
            "model_filename": "kokoro-v1_0.pth",
            "config_filename": "config.json"
        },
        "voices": {
            "use_local": True,
            "local_path": "./voices",
            "repo_id": "hexgrad/Kokoro-82M",
            "required_count": 1
        },
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": False
        },
        "audio": {
            "sample_rate": 24000,
            "default_format": "wav"
        },
        "paths": {
            "output_dir": "outputs",
            "voices_dir": "voices",
            "models_dir": "models"
        }
    }
    
    config_file = Path("tts_config_local_example.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    print(f"Example local configuration saved to {config_file}")
    print("Copy this file to tts_config.json and modify paths as needed")

def setup_local_environment():
    """Interactive setup for local environment"""
    print("=== Kokoro TTS Local Setup ===")
    print("This will help you configure local model and voice loading.")
    print()
    
    config = TTSConfig()
    
    # Model configuration
    print("Model Configuration:")
    use_local_model = input("Use local model file? (y/n, default: n): ").lower().strip()
    if use_local_model in ['y', 'yes']:
        model_path = input("Enter path to local model file (e.g., ./models/kokoro-v1_0.pth): ").strip()
        if model_path:
            config.update_config({
                "model": {
                    "use_local": True,
                    "local_path": model_path
                }
            })
            print(f"✓ Local model path set to: {model_path}")
    
    print()
    
    # Voices configuration
    print("Voices Configuration:")
    use_local_voices = input("Use local voices directory? (y/n, default: n): ").lower().strip()
    if use_local_voices in ['y', 'yes']:
        voices_path = input("Enter path to local voices directory (e.g., ./voices): ").strip()
        if voices_path:
            config.update_config({
                "voices": {
                    "use_local": True,
                    "local_path": voices_path
                }
            })
            print(f"✓ Local voices path set to: {voices_path}")
    
    print()
    
    # Server configuration
    print("Server Configuration:")
    host = input("Server host (default: 127.0.0.1): ").strip()
    if host:
        config.update_config({
            "server": {
                "host": host
            }
        })
    
    port = input("Server port (default: 5000): ").strip()
    if port and port.isdigit():
        config.update_config({
            "server": {
                "port": int(port)
            }
        })
    
    print()
    print("Configuration saved to tts_config.json")
    print("You can edit this file manually to make further changes.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            create_local_config_example()
        elif sys.argv[1] == "setup":
            setup_local_environment()
        else:
            print("Usage:")
            print("  python tts_config.py example  # Create example config")
            print("  python tts_config.py setup    # Interactive setup")
    else:
        # Test configuration
        config = TTSConfig()
        print("Current configuration:")
        print(json.dumps(config.config, indent=2))
        
        print("\nValidation:")
        validation = config.validate_local_paths()
        for key, valid in validation.items():
            status = "✓" if valid else "✗"
            print(f"  {status} {key}: {'Valid' if valid else 'Invalid'}") 
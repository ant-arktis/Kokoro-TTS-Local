#!/usr/bin/env python3
"""
Test script for the Kokoro TTS API Server
----------------------------------------
This script demonstrates how to use the TTS API endpoints.

Usage:
    python test_tts_api.py

Make sure the TTS API server is running on localhost:5000 before running this script.
"""

import requests
import json
import os
from pathlib import Path

# API server configuration
API_BASE_URL = "http://localhost:5000"
TTS_ENDPOINT = f"{API_BASE_URL}/tts/"
VOICES_ENDPOINT = f"{API_BASE_URL}/tts/voices"
HEALTH_ENDPOINT = f"{API_BASE_URL}/tts/health"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_get_voices():
    """Test getting available voices."""
    print("\n🎤 Testing get voices...")
    try:
        response = requests.get(VOICES_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            voices = data.get('voices', [])
            print(f"✅ Found {len(voices)} voices: {voices[:5]}{'...' if len(voices) > 5 else ''}")
            return voices
        else:
            print(f"❌ Get voices failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Get voices error: {e}")
        return []

def test_tts_generation(voice_name, text, speed=1.0):
    """Test TTS generation."""
    print(f"\n🎵 Testing TTS generation with voice '{voice_name}'...")
    
    # Prepare request data
    data = {
        'text': text,
        'voice': voice_name,
        'speed': speed
    }
    
    try:
        # Make POST request to TTS endpoint
        response = requests.post(TTS_ENDPOINT, json=data, stream=True)
        
        if response.status_code == 200:
            # Get response headers
            voice_used = response.headers.get('X-Voice', 'unknown')
            speed_used = response.headers.get('X-Speed', 'unknown')
            text_length = response.headers.get('X-Text-Length', 'unknown')
            
            print(f"✅ TTS generation successful!")
            print(f"   Voice used: {voice_used}")
            print(f"   Speed used: {speed_used}")
            print(f"   Text length: {text_length}")
            
            # Save audio to file
            output_dir = Path("api_outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = f"api_{voice_name}_{speed}_{len(text)}"
            output_file = output_dir / f"{timestamp}.wav"
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = output_file.stat().st_size
            print(f"   Audio saved to: {output_file} ({file_size} bytes)")
            return True
            
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            print(f"❌ TTS generation failed: {response.status_code}")
            if error_data:
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        return False

def test_form_data_tts(voice_name, text, speed=1.0):
    """Test TTS generation using form data instead of JSON."""
    print(f"\n📝 Testing TTS generation with form data...")
    
    # Prepare form data
    data = {
        'text': text,
        'voice': voice_name,
        'speed': str(speed)
    }
    
    try:
        # Make POST request with form data
        response = requests.post(TTS_ENDPOINT, data=data, stream=True)
        
        if response.status_code == 200:
            print(f"✅ Form data TTS generation successful!")
            
            # Save audio to file
            output_dir = Path("api_outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = f"form_{voice_name}_{speed}_{len(text)}"
            output_file = output_dir / f"{timestamp}.wav"
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = output_file.stat().st_size
            print(f"   Audio saved to: {output_file} ({file_size} bytes)")
            return True
            
        else:
            print(f"❌ Form data TTS generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Form data TTS generation error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Kokoro TTS API Test Script")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Health check failed. Make sure the TTS API server is running!")
        print("   Start the server with: python tts_api_server.py")
        return
    
    # Test 2: Get available voices
    voices = test_get_voices()
    if not voices:
        print("\n❌ No voices available. Check your voices directory!")
        return
    
    # Test 3: TTS generation with first available voice
    test_texts = [
        "Hello, this is a test of the Kokoro TTS API server.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer test sentence to see how the API handles different text lengths."
    ]
    
    # Test with different voices and speeds
    test_configs = [
        (voices[0], test_texts[0], 1.0),
        (voices[0], test_texts[1], 1.5),
        (voices[0], test_texts[2], 0.8),
    ]
    
    # If multiple voices are available, test with a different voice
    if len(voices) > 1:
        test_configs.append((voices[1], test_texts[0], 1.0))
    
    for voice, text, speed in test_configs:
        test_tts_generation(voice, text, speed)
    
    # Test 4: Form data TTS generation
    test_form_data_tts(voices[0], test_texts[0], 1.0)
    
    print("\n🎉 All tests completed!")
    print("Check the 'api_outputs' directory for generated audio files.")

if __name__ == "__main__":
    main() 
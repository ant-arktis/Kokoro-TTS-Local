#!/usr/bin/env python3
"""
Test script for the Kokoro TTS API Server
----------------------------------------
This script demonstrates how to use the TTS API endpoints with word timestamps.

Usage:
    python test_tts_api.py

Make sure the TTS API server is running on localhost:5000 before running this script.
"""

import requests
import json
import os
import base64
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

def test_tts_with_timestamps(voice_name, text, speed=1.0):
    """Test TTS generation with word timestamps."""
    print(f"\n🎵 Testing TTS generation with timestamps (voice: '{voice_name}')...")
    
    # Prepare request data
    data = {
        'text': text,
        'voice': voice_name,
        'speed': speed,
        'include_timestamps': True
    }
    
    try:
        # Make POST request to TTS endpoint
        response = requests.post(TTS_ENDPOINT, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            
            print(f"✅ TTS generation with timestamps successful!")
            print(f"   Voice used: {response_data.get('voice')}")
            print(f"   Speed used: {response_data.get('speed')}")
            print(f"   Sample rate: {response_data.get('sample_rate')}")
            print(f"   Total duration: {response_data.get('total_duration', 0):.2f} seconds")
            
            # Display word timestamps
            word_timestamps = response_data.get('word_timestamps', [])
            print(f"   Word timestamps: {len(word_timestamps)} words")
            
            if word_timestamps:
                print("   Word timing details:")
                for i, word_data in enumerate(word_timestamps[:10]):  # Show first 10 words
                    print(f"     {i+1:2d}. '{word_data['word']}': {word_data['start_time']:.2f}s - {word_data['end_time']:.2f}s")
                if len(word_timestamps) > 10:
                    print(f"     ... and {len(word_timestamps) - 10} more words")
            
            # Save audio to file
            output_dir = Path("api_outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = f"timestamps_{voice_name}_{speed}_{len(text)}"
            output_file = output_dir / f"{timestamp}.wav"
            
            # Decode base64 audio and save
            audio_base64 = response_data.get('audio', '')
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                file_size = output_file.stat().st_size
                print(f"   Audio saved to: {output_file} ({file_size} bytes)")
            
            # Save timestamps to JSON file
            timestamps_file = output_dir / f"{timestamp}_timestamps.json"
            with open(timestamps_file, 'w', encoding='utf-8') as f:
                json.dump(word_timestamps, f, indent=2, ensure_ascii=False)
            print(f"   Timestamps saved to: {timestamps_file}")
            
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

def test_tts_streaming(voice_name, text, speed=1.0):
    """Test TTS generation with streaming audio (no timestamps)."""
    print(f"\n🎵 Testing TTS streaming (voice: '{voice_name}')...")
    
    # Prepare request data
    data = {
        'text': text,
        'voice': voice_name,
        'speed': speed,
        'include_timestamps': False
    }
    
    try:
        # Make POST request to TTS endpoint
        response = requests.post(TTS_ENDPOINT, json=data, stream=True)
        
        if response.status_code == 200:
            # Get response headers
            voice_used = response.headers.get('X-Voice', 'unknown')
            speed_used = response.headers.get('X-Speed', 'unknown')
            text_length = response.headers.get('X-Text-Length', 'unknown')
            
            print(f"✅ TTS streaming successful!")
            print(f"   Voice used: {voice_used}")
            print(f"   Speed used: {speed_used}")
            print(f"   Text length: {text_length}")
            
            # Save audio to file
            output_dir = Path("api_outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = f"streaming_{voice_name}_{speed}_{len(text)}"
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
            print(f"❌ TTS streaming failed: {response.status_code}")
            if error_data:
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ TTS streaming error: {e}")
        return False

def test_form_data_tts(voice_name, text, speed=1.0):
    """Test TTS generation using form data instead of JSON."""
    print(f"\n📝 Testing TTS generation with form data...")
    
    # Prepare form data
    data = {
        'text': text,
        'voice': voice_name,
        'speed': str(speed),
        'include_timestamps': 'true'
    }
    
    try:
        # Make POST request with form data
        response = requests.post(TTS_ENDPOINT, data=data)
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ Form data TTS generation successful!")
            
            # Display timestamp info
            word_timestamps = response_data.get('word_timestamps', [])
            print(f"   Word timestamps: {len(word_timestamps)} words")
            
            # Save audio to file
            output_dir = Path("api_outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = f"form_{voice_name}_{speed}_{len(text)}"
            output_file = output_dir / f"{timestamp}.wav"
            
            # Decode base64 audio and save
            audio_base64 = response_data.get('audio', '')
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                file_size = output_file.stat().st_size
                print(f"   Audio saved to: {output_file} ({file_size} bytes)")
            
            return True
            
        else:
            print(f"❌ Form data TTS generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Form data TTS generation error: {e}")
        return False

def demonstrate_timestamp_usage():
    """Demonstrate how to use timestamps for UI highlighting."""
    print("\n🎯 Timestamp Usage Example:")
    print("=" * 50)
    
    example_timestamps = [
        {'word': 'Hello', 'start_time': 0.0, 'end_time': 0.5},
        {'word': 'world', 'start_time': 0.6, 'end_time': 1.2},
        {'word': 'this', 'start_time': 1.3, 'end_time': 1.7},
        {'word': 'is', 'start_time': 1.8, 'end_time': 2.0},
        {'word': 'a', 'start_time': 2.1, 'end_time': 2.2},
        {'word': 'test', 'start_time': 2.3, 'end_time': 2.8}
    ]
    
    print("Example word timestamps:")
    for word_data in example_timestamps:
        print(f"  '{word_data['word']}': {word_data['start_time']:.2f}s - {word_data['end_time']:.2f}s")
    
    print("\nJavaScript example for UI highlighting:")
    print("""
// Example JavaScript code for highlighting words during audio playback
const audio = new Audio('data:audio/wav;base64,' + response.audio);
const wordTimestamps = response.word_timestamps;

audio.addEventListener('timeupdate', () => {
    const currentTime = audio.currentTime;
    
    // Find the current word being spoken
    const currentWord = wordTimestamps.find(word => 
        currentTime >= word.start_time && currentTime <= word.end_time
    );
    
    if (currentWord) {
        // Highlight the current word in your UI
        highlightWord(currentWord.word);
    }
});

function highlightWord(word) {
    // Your UI highlighting logic here
    console.log('Highlighting word:', word);
}
    """)

def main():
    """Main test function."""
    print("🚀 Kokoro TTS API Test Script (with Timestamps)")
    print("=" * 60)
    
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
    
    # Test 3: TTS generation with timestamps
    test_texts = [
        "Hello, this is a test of the Kokoro TTS API server with word timestamps.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer test sentence to see how the API handles different text lengths and provides detailed timing information for each word."
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
        test_tts_with_timestamps(voice, text, speed)
    
    # Test 4: Streaming audio (no timestamps)
    test_tts_streaming(voices[0], test_texts[0], 1.0)
    
    # Test 5: Form data TTS generation
    test_form_data_tts(voices[0], test_texts[0], 1.0)
    
    # Demonstrate timestamp usage
    demonstrate_timestamp_usage()
    
    print("\n🎉 All tests completed!")
    print("Check the 'api_outputs' directory for generated audio files and timestamp JSON files.")
    print("\n📝 Usage Notes:")
    print("- Set 'include_timestamps: true' to get word-level timing data")
    print("- Set 'include_timestamps: false' for streaming audio only")
    print("- Timestamps are in seconds and can be used for UI highlighting")
    print("- Audio is returned as base64-encoded WAV when timestamps are included")

if __name__ == "__main__":
    main() 
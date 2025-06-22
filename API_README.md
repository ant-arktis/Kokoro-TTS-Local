# Kokoro TTS API Server

A Flask-based API server that provides streaming TTS audio generation using the Kokoro TTS package. This server can be used as a backend for web applications, mobile apps, or any other service that needs text-to-speech functionality.

## Features

- **RESTful API**: Clean HTTP endpoints for TTS generation
- **Streaming Audio**: Real-time audio streaming for immediate playback
- **Word-Level Timestamps**: Precise timing data for UI highlighting
- **Multiple Voices**: Support for all available Kokoro voices
- **Configurable Speed**: Adjustable speech speed (0.1x to 3.0x)
- **Health Monitoring**: Built-in health check endpoint
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Nginx Ready**: Designed to work with nginx reverse proxy

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python tts_api_server.py
```

The server will start on `http://127.0.0.1:5000` by default.

### 3. Test the API

```bash
python test_tts_api.py
```

## API Endpoints

### POST `/tts/`
Generate streaming TTS audio with optional word timestamps.

**Request Body (JSON):**
```json
{
    "text": "Hello, this is a test of the TTS API.",
    "voice": "af_bella",
    "speed": 1.0,
    "format": "wav",
    "include_timestamps": true
}
```

**Request Body (Form Data):**
```
text=Hello, this is a test of the TTS API.
voice=af_bella
speed=1.0
format=wav
include_timestamps=true
```

**Response (with timestamps):**
```json
{
    "audio": "base64_encoded_wav_audio",
    "audio_format": "wav",
    "sample_rate": 24000,
    "word_timestamps": [
        {
            "word": "Hello",
            "start_time": 0.0,
            "end_time": 0.5,
            "phonemes": "həˈloʊ",
            "whitespace": " "
        },
        {
            "word": "this",
            "start_time": 0.6,
            "end_time": 1.1,
            "phonemes": "ðɪs",
            "whitespace": " "
        }
    ],
    "total_duration": 3.2,
    "voice": "af_bella",
    "speed": 1.0,
    "text": "Hello, this is a test of the TTS API."
}
```

**Response (streaming audio):**
- **Content-Type**: `audio/wav`
- **Headers**: 
  - `X-Voice`: Voice used
  - `X-Speed`: Speed used
  - `X-Text-Length`: Length of input text
  - `X-Include-Timestamps`: Whether timestamps were included
- **Body**: Streaming WAV audio data

### GET `/tts/voices`
Get list of available voices.

**Response:**
```json
{
    "voices": ["af_bella", "af_jessica", "am_john", ...],
    "count": 54
}
```

### GET `/tts/health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "voices_available": 54,
    "device": "cuda",
    "model_loaded": true
}
```

### GET `/tts/`
Get API information.

**Response:**
```json
{
    "service": "Kokoro TTS API",
    "version": "1.0.0",
    "endpoints": {
        "POST /tts/": "Generate streaming TTS audio with timestamps",
        "GET /tts/voices": "Get available voices",
        "GET /tts/health": "Health check"
    },
    "parameters": {
        "text": "Text to convert to speech (required)",
        "voice": "Voice name (required)",
        "speed": "Speech speed (0.1-3.0, default: 1.0)",
        "format": "Output format (wav, mp3, aac, default: wav)",
        "include_timestamps": "Include word timestamps (true/false, default: true)"
    },
    "features": {
        "word_timestamps": "Word-level timing for UI highlighting",
        "streaming_audio": "Real-time audio streaming",
        "multiple_voices": "Support for 54+ voices across 8 languages"
    }
}
```

## Configuration

### Environment Variables

- `TTS_HOST`: Server host (default: `127.0.0.1`)
- `TTS_PORT`: Server port (default: `5000`)
- `TTS_DEBUG`: Enable debug mode (default: `false`)

### Example Usage

```bash
# Start server on different host/port
TTS_HOST=0.0.0.0 TTS_PORT=8080 python tts_api_server.py

# Enable debug mode
TTS_DEBUG=true python tts_api_server.py
```

## Nginx Integration

### 1. Copy the nginx configuration

```bash
sudo cp nginx_tts_config.conf /etc/nginx/sites-available/tts-api
```

### 2. Edit the configuration

Modify `nginx_tts_config.conf` to match your domain and requirements.

### 3. Enable the site

```bash
sudo ln -s /etc/nginx/sites-available/tts-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Test the integration

```bash
# Test through nginx
curl -X POST http://your-domain.com/tts/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella"}' \
  --output test.wav
```

## Client Examples

### JavaScript (fetch) - With Timestamps

```javascript
// Generate TTS audio with word timestamps
const response = await fetch('http://localhost:5000/tts/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'Hello, this is a test.',
        voice: 'af_bella',
        speed: 1.0,
        include_timestamps: true
    })
});

const data = await response.json();

// Create audio from base64
const audioBlob = new Blob([
    Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))
], { type: 'audio/wav' });

const audio = new Audio(URL.createObjectURL(audioBlob));

// Highlight words during playback
audio.addEventListener('timeupdate', () => {
    const currentTime = audio.currentTime;
    
    // Find the current word being spoken
    const currentWord = data.word_timestamps.find(word => 
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

// Play the audio
audio.play();
```

### JavaScript (fetch) - Streaming Audio

```javascript
// Generate streaming TTS audio (no timestamps)
const response = await fetch('http://localhost:5000/tts/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'Hello, this is a test.',
        voice: 'af_bella',
        speed: 1.0,
        include_timestamps: false
    }),
    stream: true
});

// Play audio directly
const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

### cURL - With Timestamps

```bash
# Generate TTS audio with word timestamps
curl -X POST http://localhost:5000/tts/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "af_bella",
    "speed": 1.0,
    "include_timestamps": true
  }' \
  --output response.json

# Extract audio from JSON response
jq -r '.audio' response.json | base64 -d > audio.wav

# Display word timestamps
jq '.word_timestamps[] | "\(.word): \(.start_time)s - \(.end_time)s"' response.json
```

### cURL - Streaming Audio

```bash
# Generate streaming TTS audio (no timestamps)
curl -X POST http://localhost:5000/tts/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "af_bella",
    "speed": 1.0,
    "include_timestamps": false
  }' \
  --output output.wav
```

### cURL - Form Data

```bash
# Generate TTS using form data
curl -X POST http://localhost:5000/tts/ \
  -d "text=Hello world" \
  -d "voice=af_bella" \
  -d "speed=1.0" \
  -d "include_timestamps=true" \
  --output response.json
```

### Python (requests) - With Timestamps

```python
import requests
import base64
import json

# Generate TTS audio with timestamps
response = requests.post(
    'http://localhost:5000/tts/',
    json={
        'text': 'Hello, this is a test.',
        'voice': 'af_bella',
        'speed': 1.0,
        'include_timestamps': True
    }
)

data = response.json()

# Save audio to file
audio_data = base64.b64decode(data['audio'])
with open('output.wav', 'wb') as f:
    f.write(audio_data)

# Print word timestamps
for word_data in data['word_timestamps']:
    print(f"'{word_data['word']}': {word_data['start_time']:.2f}s - {word_data['end_time']:.2f}s")
```

### Python (requests) - Streaming

```python
import requests

# Generate streaming TTS audio
response = requests.post(
    'http://localhost:5000/tts/',
    json={
        'text': 'Hello, this is a test.',
        'voice': 'af_bella',
        'speed': 1.0,
        'include_timestamps': False
    },
    stream=True
)

# Save audio to file
with open('output.wav', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

## Word Timestamp Format

Each word timestamp contains:

```json
{
    "word": "Hello",
    "start_time": 0.0,
    "end_time": 0.5,
    "phonemes": "həˈloʊ",
    "whitespace": " "
}
```

- **word**: The actual word text
- **start_time**: When the word starts (in seconds)
- **end_time**: When the word ends (in seconds)
- **phonemes**: Phonetic representation (if available)
- **whitespace**: Space character after the word (if any)

## UI Integration Examples

### React Component Example

```jsx
import React, { useState, useEffect } from 'react';

function TTSPlayer({ text, voice }) {
    const [audio, setAudio] = useState(null);
    const [timestamps, setTimestamps] = useState([]);
    const [currentWord, setCurrentWord] = useState(null);

    const generateTTS = async () => {
        const response = await fetch('/tts/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                voice,
                include_timestamps: true
            })
        });

        const data = await response.json();
        
        // Create audio
        const audioBlob = new Blob([
            Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))
        ], { type: 'audio/wav' });
        
        const audioUrl = URL.createObjectURL(audioBlob);
        const newAudio = new Audio(audioUrl);
        
        // Setup word highlighting
        newAudio.addEventListener('timeupdate', () => {
            const currentTime = newAudio.currentTime;
            const word = data.word_timestamps.find(w => 
                currentTime >= w.start_time && currentTime <= w.end_time
            );
            setCurrentWord(word?.word || null);
        });

        setAudio(newAudio);
        setTimestamps(data.word_timestamps);
    };

    return (
        <div>
            <button onClick={generateTTS}>Generate Speech</button>
            <button onClick={() => audio?.play()}>Play</button>
            <button onClick={() => audio?.pause()}>Pause</button>
            
            <div className="text-display">
                {text.split(' ').map((word, index) => (
                    <span
                        key={index}
                        className={currentWord === word ? 'highlighted' : ''}
                    >
                        {word}{' '}
                    </span>
                ))}
            </div>
        </div>
    );
}
```

### Vue.js Component Example

```vue
<template>
    <div>
        <button @click="generateTTS">Generate Speech</button>
        <button @click="playAudio" :disabled="!audio">Play</button>
        <button @click="pauseAudio" :disabled="!audio">Pause</button>
        
        <div class="text-display">
            <span
                v-for="(word, index) in words"
                :key="index"
                :class="{ highlighted: currentWord === word }"
            >
                {{ word }}
            </span>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            audio: null,
            timestamps: [],
            currentWord: null,
            words: []
        };
    },
    methods: {
        async generateTTS() {
            const response = await fetch('/tts/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: this.text,
                    voice: this.voice,
                    include_timestamps: true
                })
            });

            const data = await response.json();
            
            // Create audio
            const audioBlob = new Blob([
                Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))
            ], { type: 'audio/wav' });
            
            this.audio = new Audio(URL.createObjectURL(audioBlob));
            this.timestamps = data.word_timestamps;
            this.words = this.text.split(' ');
            
            // Setup word highlighting
            this.audio.addEventListener('timeupdate', () => {
                const currentTime = this.audio.currentTime;
                const word = this.timestamps.find(w => 
                    currentTime >= w.start_time && currentTime <= w.end_time
                );
                this.currentWord = word?.word || null;
            });
        },
        
        playAudio() {
            this.audio?.play();
        },
        
        pauseAudio() {
            this.audio?.pause();
        }
    }
};
</script>

<style scoped>
.highlighted {
    background-color: yellow;
    font-weight: bold;
}
</style>
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid parameters (missing text, invalid voice, etc.)
- `500 Internal Server Error`: Server-side errors (model loading, audio generation, etc.)

Error responses include a JSON object with an `error` field:

```json
{
    "error": "Voice 'invalid_voice' not found. Available voices: ['af_bella', 'af_jessica', ...]"
}
```

## Performance Considerations

- **Memory Usage**: The server loads the TTS model into memory on startup
- **Concurrent Requests**: Flask's threaded mode handles multiple concurrent requests
- **Audio Streaming**: Large audio files are streamed to reduce memory usage
- **Voice Caching**: Voices are cached in memory after first use
- **Timestamp Generation**: Word timestamps add minimal overhead to audio generation

## Troubleshooting

### Common Issues

1. **No voices available**
   - Check that the `voices/` directory exists and contains `.pt` files
   - Run the voice download script if needed

2. **CUDA out of memory**
   - Reduce the number of concurrent requests
   - Use CPU mode by setting `device='cpu'` in the code

3. **Nginx proxy issues**
   - Check nginx error logs: `sudo tail -f /var/log/nginx/error.log`
   - Verify the Flask server is running on the correct port
   - Test direct access to the Flask server first

4. **Audio quality issues**
   - Ensure the sample rate is set correctly (default: 24000 Hz)
   - Check that the voice files are not corrupted

5. **Timestamp accuracy**
   - Timestamps are based on the TTS model's internal timing
   - May vary slightly from actual audio playback due to processing delays
   - For precise synchronization, use the `timeupdate` event with small intervals

### Logs

The server logs important events to stdout. For production, consider redirecting logs to files:

```bash
python tts_api_server.py > tts.log 2>&1
```

## Security Considerations

- The API server is designed for local/trusted network use
- For public deployment, consider:
  - Adding authentication/authorization
  - Rate limiting
  - Input validation and sanitization
  - HTTPS/TLS encryption
  - Firewall rules

## License

This API server is part of the Kokoro TTS Local project and follows the same license terms. 
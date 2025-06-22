# Kokoro TTS API Server

A Flask-based API server that provides streaming TTS audio generation using the Kokoro TTS package. This server can be used as a backend for web applications, mobile apps, or any other service that needs text-to-speech functionality.

## Features

- **RESTful API**: Clean HTTP endpoints for TTS generation
- **Streaming Audio**: Real-time audio streaming for immediate playback
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
Generate streaming TTS audio.

**Request Body (JSON):**
```json
{
    "text": "Hello, this is a test of the TTS API.",
    "voice": "af_bella",
    "speed": 1.0,
    "format": "wav"
}
```

**Request Body (Form Data):**
```
text=Hello, this is a test of the TTS API.
voice=af_bella
speed=1.0
format=wav
```

**Response:**
- **Content-Type**: `audio/wav`
- **Headers**: 
  - `X-Voice`: Voice used
  - `X-Speed`: Speed used
  - `X-Text-Length`: Length of input text
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
        "POST /tts/": "Generate streaming TTS audio",
        "GET /tts/voices": "Get available voices",
        "GET /tts/health": "Health check"
    },
    "parameters": {
        "text": "Text to convert to speech (required)",
        "voice": "Voice name (required)",
        "speed": "Speech speed (0.1-3.0, default: 1.0)",
        "format": "Output format (wav, mp3, aac, default: wav)"
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

### Python (requests)

```python
import requests

# Generate TTS audio
response = requests.post(
    'http://localhost:5000/tts/',
    json={
        'text': 'Hello, this is a test.',
        'voice': 'af_bella',
        'speed': 1.0
    },
    stream=True
)

# Save audio to file
with open('output.wav', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

### JavaScript (fetch)

```javascript
// Generate TTS audio
const response = await fetch('http://localhost:5000/tts/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'Hello, this is a test.',
        voice: 'af_bella',
        speed: 1.0
    })
});

// Play audio directly
const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

### cURL

```bash
# Generate TTS audio
curl -X POST http://localhost:5000/tts/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella", "speed": 1.0}' \
  --output output.wav

# Get available voices
curl http://localhost:5000/tts/voices

# Health check
curl http://localhost:5000/tts/health
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
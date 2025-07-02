# AI Music Video Creator

A Flask web application that creates AI-generated music videos using advanced machine learning APIs.

## Features

- 🎵 **AI Music Generation**: Create custom music tracks using Suno-compatible tags
- 🎨 **Dynamic Visuals**: Generate colorful, animated backgrounds based on your vision
- 🤖 **Gemini AI Integration**: Intelligent tag generation for music styles
- 🎬 **Automated Video Assembly**: Combines audio and visuals into polished videos
- ☁️ **Cloud Storage**: Upload and share videos via Google Cloud Storage
- 📱 **Responsive Design**: Beautiful, modern interface that works on all devices

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment** (Optional)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   ```
   http://localhost:5000
   ```

## Demo Mode

The app works in demo mode without any API keys! It will:
- Generate placeholder music
- Create colorful visual backgrounds
- Assemble a sample video
- Provide a demo download link

## API Configuration

### Required for Full Functionality

- **GEMINI_API_KEY**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **APIBOX_KEY**: Music generation service API key
- **GCS_BUCKET_NAME**: Google Cloud Storage bucket (optional)

### Environment Variables

```bash
# Core APIs
GEMINI_API_KEY=your_gemini_api_key
APIBOX_KEY=your_music_api_key

# Storage (optional)
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Deployment
PORT=5000
```

## Video Creation Process

1. **Form Input**: User selects mood, age, quality, length, and optional artist/vision
2. **AI Tag Generation**: Gemini AI creates Suno-compatible music tags
3. **Music Generation**: API creates custom audio track
4. **Visual Creation**: Generate colorful, dynamic backgrounds
5. **Video Assembly**: Combine audio and visuals using MoviePy
6. **Cloud Upload**: Store video in Google Cloud Storage
7. **Delivery**: Provide shareable video URL

## Dependencies

- **Flask**: Web framework
- **MoviePy**: Video processing
- **Librosa**: Audio analysis
- **Pillow**: Image generation
- **NumPy**: Numerical processing
- **Requests**: HTTP client
- **Google Cloud Storage**: Cloud storage (optional)

## Deployment

The app includes Railway deployment configuration:

```json
{
  "build": { "builder": "NIXPACKS" },
  "deploy": { "startCommand": "gunicorn app:app" }
}
```

## Error Handling

The application includes comprehensive error handling:
- API failures fall back to demo mode
- Missing dependencies use fallback implementations
- Network issues are handled gracefully
- User-friendly error messages

## Security

- Environment variables for sensitive data
- Input validation and sanitization
- Secure file handling
- No hardcoded credentials

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify your API keys are correct
3. Ensure all dependencies are installed
4. Try running in demo mode first

---

**Built with ❤️ using AI and modern web technologies**
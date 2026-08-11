# SignalCheck

A Chrome extension + FastAPI backend that evaluates text authenticity and domain reputation, providing a "Signal Trust Score".

## Local Development

### Backend

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Google API key:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. Start the server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

4. Test the API:
   ```bash
   curl -X POST http://localhost:8000/api/scan \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com", "text_snippet": "Sample text to analyze"}'
   ```

### Chrome Extension

1. Open Chrome and navigate to `chrome://extensions`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The SignalCheck icon will appear in your toolbar

## Deployment (Render)

### Deploy Backend

1. Push code to GitHub
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` configuration
5. Add environment variable: `GOOGLE_API_KEY` = your API key
6. Deploy

### Update Extension for Production

1. Edit `extension/config.js`:
   ```javascript
   const CONFIG = {
     API_URL: "https://your-app-name.onrender.com/api/scan",
     IS_PRODUCTION: true,
   };
   ```

2. Reload the extension in `chrome://extensions`

## Signal Trust Score

The combined score (0-100) is calculated from:
- **Domain Signal Score (40%)**: Reputation of the website domain
- **Content Authenticity (60%)**: Likelihood the content is human-written (inverse of AI probability)

Higher scores indicate more trustworthy content.

## Architecture

```
┌─────────────────┐     POST /api/scan     ┌─────────────────┐
│ Chrome Extension│ ──────────────────────▶│  FastAPI Backend│
│                 │                        │    (Render)     │
│ - popup.html    │                        │                 │
│ - config.js     │                        │ - Domain check  │
│ - popup.js      │◀─────────────────────  │ - Gemini API    │
│ - content.js    │   Signal Trust Score   │   analysis      │
└─────────────────┘                        └─────────────────┘
```

## Tech Stack

- **Backend**: FastAPI (Python)
- **AI**: Google Gemini 3.6 Flash via `google-genai` SDK
- **Frontend**: Chrome Extension (Manifest V3)
- **Deployment**: Render

## Analytics

Monitor API usage and costs in [Google AI Studio](https://aistudio.google.com) dashboard.

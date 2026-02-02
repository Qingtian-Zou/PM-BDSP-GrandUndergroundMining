# Grand Underground Mining - Backend API

Flask-based REST API for the Grand Underground Mining game, designed for deployment on Railway.

## Features

- REST API for game state management
- AI agent integration for hints and competition mode
- CORS support for cross-origin requests from GitHub Pages

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

The server will start on `http://localhost:5000`.

## Deployment on Railway

### 1. Create a new Railway project
1. Go to [Railway](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this repository
4. Set the root directory to `backend`

### 2. Configure Environment Variables
In Railway dashboard, set the following environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `https://yourusername.github.io` |
| `FRONTEND_URL` | Your GitHub Pages URL | `https://yourusername.github.io/PM-BDSP-GrandUnderground-mining` |

### 3. Add Checkpoint Files
The AI model checkpoint needs to be included. Upload the `checkpoints/` folder with the trained model.

## API Endpoints

### Health Check
- `GET /` - Service health check
- `GET /api/health` - API health check

### Normal Mode
- `POST /api/game/new` - Create a new game
- `GET /api/game/<game_id>/state` - Get game state
- `POST /api/game/<game_id>/action` - Take an action
- `GET /api/game/<game_id>/suggest` - Get AI suggestion

### Compete Mode
- `POST /api/compete/new` - Create a compete session
- `GET /api/compete/<session_id>/state` - Get session state
- `POST /api/compete/<session_id>/player-action` - Player action
- `POST /api/compete/<session_id>/ai-action` - AI takes one action
- `POST /api/compete/<session_id>/ai-fast-forward` - AI completes game

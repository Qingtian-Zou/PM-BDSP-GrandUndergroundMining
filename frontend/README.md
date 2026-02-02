# Grand Underground Mining - Frontend

Static web frontend for the Grand Underground Mining game, designed for deployment on GitHub Pages.

## Features

- Beautiful, responsive game UI
- Normal mode with AI hints
- Compete mode against AI
- Pure HTML/CSS/JS - no build step required

## Local Development

You can serve the frontend locally using any static file server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js (npx)
npx serve .
```

Then open `http://localhost:8000` in your browser.

**Note:** For local development with the backend, you need to:
1. Run the backend server: `cd ../backend && python server.py`
2. Keep `API_BASE_URL` empty in `js/config.js` (default)

## Deployment on GitHub Pages

### 1. Configure the API URL

Before deploying, update `js/config.js` with your Railway backend URL:

```javascript
const API_BASE_URL = 'https://your-backend-name.up.railway.app';
```

### 2. Deploy to GitHub Pages

#### Option A: Using GitHub Actions (Recommended)
Create `.github/workflows/deploy-frontend.yml` in your repository:

```yaml
name: Deploy Frontend to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./frontend
```

#### Option B: Manual Deployment
1. Go to your repository Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select the branch containing the `frontend` folder
4. If your frontend is in a subdirectory, you may need to use GitHub Actions

### 3. Access Your Game

After deployment, your game will be available at:
```
https://yourusername.github.io/PM-BDSP-GrandUnderground-mining/
```

## File Structure

```
frontend/
├── index.html      # Main HTML file
├── css/
│   └── style.css   # Game styles
├── js/
│   ├── config.js   # API configuration
│   ├── game.js     # Main game logic
│   └── ai-assist.js # AI hint functionality
└── assets/         # Game assets
```

## Configuration

### `js/config.js`

```javascript
// For local development with local backend
const API_BASE_URL = '';

// For production with Railway backend
const API_BASE_URL = 'https://your-app.up.railway.app';
```

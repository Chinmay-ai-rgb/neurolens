# Deployment Guide for Streamlit Community Cloud

## Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free at https://streamlit.io/cloud)
3. Git installed locally

## Step 1: Initialize Git Repository

```bash
cd neurolens
git init
git add .
git commit -m "Initial commit: NeuroLens+ system"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `neurolens`)
3. **DO NOT** initialize with README, .gitignore, or license
4. Copy the repository URL

## Step 3: Connect Local Repository to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/neurolens.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Step 4: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/neurolens`
5. **Main file path**: Enter `interface/app.py`
6. **Python version**: Select 3.10 or higher
7. Click "Deploy!"

## Important Notes

### ⚠️ Webcam Limitations

**Streamlit Community Cloud does not support direct webcam access**. The eye tracking tasks require a webcam, so:

- **Option 1**: Use locally (recommended for testing):
  ```bash
  streamlit run interface/app.py
  ```

- **Option 2**: For cloud deployment, you would need to:
  - Upload pre-recorded video files, OR
  - Use a different deployment platform that supports webcams (like Heroku with buildpacks, or your own server)

### Alternative: Local Deployment

For full functionality with webcam access, run locally:

```bash
pip install -r requirements.txt
streamlit run interface/app.py
```

## Troubleshooting

### If deployment fails:

1. **Check requirements.txt**: Ensure all dependencies are compatible with Streamlit Cloud
2. **Check Python version**: Streamlit Cloud supports Python 3.8-3.11
3. **Check file paths**: Make sure `interface/app.py` exists and is the correct entry point
4. **Check imports**: All imports should work in the cloud environment

### Common Issues:

- **ModuleNotFoundError**: Add missing packages to `requirements.txt`
- **Port already in use**: Streamlit Cloud handles this automatically
- **Webcam not found**: This is expected on Streamlit Cloud - webcams are not accessible

## Environment Variables (if needed)

If you need environment variables:
1. In Streamlit Cloud dashboard, go to "Settings"
2. Add secrets or environment variables as needed

## Updating Your App

After making changes:

```bash
git add .
git commit -m "Update: [describe changes]"
git push origin main
```

Streamlit Cloud will automatically redeploy your app!


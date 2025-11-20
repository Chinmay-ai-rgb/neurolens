# 🚀 Quick Deployment Guide

## Current Status: ✅ Git Repository Initialized

Your code is ready to be pushed to GitHub and deployed to Streamlit Community Cloud.

## ⚠️ CRITICAL: Webcam Limitation

**Streamlit Community Cloud does NOT support webcam access.**

- The app will deploy, but **eye tracking tasks won't work**
- You'll see the UI, but video capture will fail
- **For full functionality, run locally** (see below)

## Next Steps to Deploy

### 1. Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `neurolens` (or your choice)
3. **Important**: Leave README, .gitignore, and license **UNCHECKED**
4. Click "Create repository"

### 2. Connect and Push

```bash
cd /Users/vikramgujar/neurolens

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/neurolens.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Settings:
   - **Repository**: `YOUR_USERNAME/neurolens`
   - **Branch**: `main`
   - **Main file path**: `interface/app.py`
   - **Python version**: `3.10` or `3.11`
5. Click "Deploy!"

### 4. Wait for Deployment

- First deployment takes 2-5 minutes
- Check deployment logs for any errors
- Once deployed, you'll get a URL like: `https://neurolens.streamlit.app`

## 🔧 Alternative: Run Locally (Recommended)

For **full functionality** with webcam access:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run interface/app.py
```

The app will open at `http://localhost:8501` with full webcam support.

## 📝 Files Already Created

✅ `.gitignore` - Ignores unnecessary files  
✅ `.streamlit/config.toml` - Streamlit configuration  
✅ `requirements.txt` - Python dependencies  
✅ Git repository initialized and first commit created  

## ❓ Need Help?

- Check `DEPLOYMENT.md` for detailed instructions
- Check `STREAMLIT_CLOUD_NOTE.md` for webcam limitation details
- Check `README.md` for full documentation


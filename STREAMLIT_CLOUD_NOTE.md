# ⚠️ Important: Streamlit Community Cloud Limitations

## Webcam Access Issue

**Streamlit Community Cloud does NOT support direct webcam access.**

This NeuroLens+ application requires:
- Real-time webcam access for eye tracking
- MediaPipe FaceMesh for pupil detection
- OpenCV for video capture

### What This Means

1. **You CAN deploy to Streamlit Cloud**, but:
   - The eye tracking tasks **will not work** (no webcam access)
   - The UI will load, but video capture will fail
   - Feature extraction from uploaded files might work (if implemented)

2. **For full functionality**, you must run locally:
   ```bash
   pip install -r requirements.txt
   streamlit run interface/app.py
   ```

### Alternatives for Cloud Deployment

1. **Use a different platform** that supports webcams:
   - Heroku (with custom buildpacks)
   - Your own VPS/server
   - AWS EC2 / Google Cloud / Azure

2. **Modify the app** to accept:
   - Pre-recorded video file uploads
   - Video file processing instead of live webcam

3. **Use local deployment** for actual testing and use cases

### Current Deployment Steps (Limited Functionality)

If you still want to deploy to Streamlit Cloud for UI testing:

1. Initialize Git (already done if you ran `setup_git.sh`)
2. Create GitHub repository
3. Push code to GitHub
4. Deploy on Streamlit Cloud with main file: `interface/app.py`
5. **Note**: Webcam tasks will not function

### Recommended: Local Development

For actual use, run locally with webcam access:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run interface/app.py
```

The app will open in your browser at `http://localhost:8501` with full webcam functionality.


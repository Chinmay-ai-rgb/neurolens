#!/bin/bash

echo "🚀 NeuroLens+ Git Setup Script"
echo "=============================="
echo ""

if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "⚠️  Git repository already exists"
fi

echo ""
echo "Adding files to Git..."
git add .

echo ""
echo "Creating initial commit..."
git commit -m "Initial commit: NeuroLens+ neurological biomarker assessment system"

echo ""
echo "✅ Git setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub (https://github.com/new)"
echo "2. Run these commands (replace YOUR_USERNAME):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/neurolens.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Then deploy on Streamlit Community Cloud:"
echo "   https://share.streamlit.io/"
echo ""
echo "⚠️  NOTE: Streamlit Cloud does NOT support webcam access."
echo "   For full functionality, run locally: streamlit run interface/app.py"


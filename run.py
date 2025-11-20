#!/usr/bin/env python3

import sys
import subprocess
import os

def main():
    print("NeuroLens+ - Starting Streamlit Interface...")
    print("\nMake sure you have installed all requirements:")
    print("  pip install -r requirements.txt\n")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "interface/app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry running manually:")
        print("  streamlit run interface/app.py")

if __name__ == "__main__":
    main()


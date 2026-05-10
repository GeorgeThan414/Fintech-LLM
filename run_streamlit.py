"""
Launch the Streamlit dashboard with optional public URL via ngrok.

Usage:
    python run_streamlit.py            # local only (http://localhost:8501)
    python run_streamlit.py --public   # also expose via ngrok public URL

Requires: pip install streamlit pyngrok
For --public, set NGROK_AUTHTOKEN in .env (get free token at https://dashboard.ngrok.com)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PORT = 8501
APP_FILE = Path(__file__).parent / "streamlit_app.py"


def start_streamlit():
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP_FILE),
        "--server.port", str(PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.Popen(cmd)


def start_ngrok():
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("ERROR: pyngrok not installed. Run: pip install pyngrok")
        sys.exit(1)

    token = os.environ.get("NGROK_AUTHTOKEN", "")
    if token:
        conf.get_default().auth_token = token
    else:
        print("WARNING: NGROK_AUTHTOKEN not set. ngrok session will be limited.")
        print("Get a free token at https://dashboard.ngrok.com and add to .env")

    public_url = ngrok.connect(PORT, "http").public_url
    return public_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true", help="Expose via ngrok")
    args = parser.parse_args()

    print(f"Starting Streamlit on port {PORT}...")
    proc = start_streamlit()
    time.sleep(4)

    if args.public:
        try:
            url = start_ngrok()
            print("\n" + "=" * 60)
            print(f"PUBLIC URL: {url}")
            print(f"LOCAL URL:  http://localhost:{PORT}")
            print("=" * 60 + "\n")
            print("Share the PUBLIC URL with your colleague.")
            print("Press CTRL+C to stop both Streamlit and ngrok.\n")
        except Exception as e:
            print(f"ngrok failed: {e}")
            print(f"Streamlit still running locally at http://localhost:{PORT}")
    else:
        print(f"\nLOCAL URL: http://localhost:{PORT}")
        print("Run with --public to also expose externally via ngrok.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        if args.public:
            try:
                from pyngrok import ngrok
                ngrok.disconnect_all()
                ngrok.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()

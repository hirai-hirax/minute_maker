"""
Manual download script for SpeechBrain speaker recognition model.
This script downloads the spkrec-ecapa-voxceleb model from Hugging Face
and saves it to the backend/tmp_model directory.
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Determine paths
script_dir = Path(__file__).parent
model_dir = script_dir / "backend" / "tmp_model"

model_id = "speechbrain/spkrec-ecapa-voxceleb"

print(f"Downloading SpeechBrain model: {model_id}")
print(f"Destination: {model_dir}")
print("This may take several minutes (~100MB download)...")
print()

try:
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the model
    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        repo_type="model"
    )
    
    print()
    print(f"✓ Model downloaded successfully to: {model_dir}")
    print()
    print("Downloaded files:")
    for file in sorted(model_dir.glob("**/*")):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.relative_to(model_dir)} ({size_mb:.2f} MB)")
    
    print()
    print("You can now start/restart the backend server.")
    
except Exception as e:
    print(f"✗ Download failed: {e}", file=sys.stderr)
    print()
    print("Troubleshooting:")
    print("1. Check your internet connection")
    print("2. Verify you can access huggingface.co")
    print("3. Try running: pip install --upgrade huggingface-hub")
    sys.exit(1)

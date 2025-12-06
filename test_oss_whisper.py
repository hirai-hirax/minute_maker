"""
Quick verification script to test OSS Whisper integration.
This script checks if faster-whisper can be loaded successfully.
"""
import os
import sys

# Set environment for testing
os.environ["WHISPER_PROVIDER"] = "faster-whisper"
os.environ["OSS_WHISPER_MODEL"] = "base"
os.environ["OSS_WHISPER_DEVICE"] = "cpu"

print("Testing OSS Whisper Integration...")
print(f"WHISPER_PROVIDER: {os.environ.get('WHISPER_PROVIDER')}")
print(f"OSS_WHISPER_MODEL: {os.environ.get('OSS_WHISPER_MODEL')}")
print(f"OSS_WHISPER_DEVICE: {os.environ.get('OSS_WHISPER_DEVICE')}")
print()

try:
    print("Importing faster-whisper...")
    from faster_whisper import WhisperModel
    print("✓ faster-whisper imported successfully")
    
    print("\nLoading Whisper model (this may take a minute on first run)...")
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )
    print("✓ Whisper model loaded successfully")
    print(f"  Model type: {type(model)}")
    
    print("\n✅ OSS Whisper integration test PASSED!")
    print("\nThe faster-whisper library is working correctly.")
    print("You can now use WHISPER_PROVIDER=faster-whisper in your .env file.")
    
except Exception as e:
    print(f"\n❌ OSS Whisper integration test FAILED")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

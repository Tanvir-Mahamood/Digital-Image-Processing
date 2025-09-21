import importlib
import subprocess
import sys

packages = ["numpy", "opencv-python", "matplotlib", "scikit-image", "Pillow"]

for pkg in packages:
    try:
        importlib.import_module(pkg.split("-")[0])  # opencv-python → cv2
        print(f"✅ {pkg} is installed")
    except ImportError:
        print(f"❌ {pkg} is missing → installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

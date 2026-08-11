import cv2
from pathlib import Path



# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Model paths
PROTOTXT_PATH = BASE_DIR / "models" / "deploy.prototxt"
MODEL_PATH = BASE_DIR / "models" / "mobilenet_iter_73000.caffemodel"


# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe(
    str(PROTOTXT_PATH),
    str(MODEL_PATH)
)

print("MobileNet-SSD model loaded successfully!")


# =========================
# Image path
# =========================

IMAGE_PATH = BASE_DIR / "images" / "cat.jpg"


# =========================
# Load image
# =========================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Image loaded successfully!")
print("Image shape:", image.shape)

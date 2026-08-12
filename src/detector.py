import cv2
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Model paths
# ============================================================

PROTOTXT_PATH = BASE_DIR / "models" / "deploy.prototxt"
MODEL_PATH = BASE_DIR / "models" / "mobilenet_iter_73000.caffemodel"


# ============================================================
# Class labels
# MobileNet-SSD was trained on these 20 object classes
# ============================================================

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


# ============================================================
# Load MobileNet-SSD model
# ============================================================

net = cv2.dnn.readNetFromCaffe(
    str(PROTOTXT_PATH),
    str(MODEL_PATH)
)

print("MobileNet-SSD model loaded successfully!")


# ============================================================
# Image path
# ============================================================

IMAGE_PATH = BASE_DIR / "images" / "cat.jpg"


# ============================================================
# Load image
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Image loaded successfully!")
print("Image shape:", image.shape)


# ============================================================
# Convert image into a blob
# ============================================================

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1 / 127.5,
    size=(300, 300),
    mean=(127.5, 127.5, 127.5),
    swapRB=True,
    crop=False
)

print("Blob shape:", blob.shape)


# ============================================================
# Run inference
# ============================================================

net.setInput(blob)

detections = net.forward()

print("Detection shape:", detections.shape)


# ============================================================
# Process detections
# ============================================================

CONFIDENCE_THRESHOLD = 0.5


for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > CONFIDENCE_THRESHOLD:

        # Get class ID
        class_id = int(detections[0, 0, i, 1])

        # Convert class ID to label
        label = CLASSES[class_id]

        # Get normalized bounding-box coordinates
        box = detections[0, 0, i, 3:7]

        # Get image dimensions
        (h, w) = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        startX = int(box[0] * w)
        startY = int(box[1] * h)

        endX = int(box[2] * w)
        endY = int(box[3] * h)

        # Draw bounding box
        cv2.rectangle(
            image,
            (startX, startY),
            (endX, endY),
            (0, 255, 0),
            2
        )

        # Create label
        text = f"{label}: {confidence * 100:.2f}%"

        # Draw label
        cv2.putText(
            image,
            text,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        print(
            f"Detection {i}: "
            f"{label} "
            f"Confidence = {confidence:.2f}"
        )

        print(
            f"Bounding Box: "
            f"({startX}, {startY}) → "
            f"({endX}, {endY})"
        )
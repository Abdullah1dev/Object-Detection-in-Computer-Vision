import cv2
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROTOTXT_PATH = BASE_DIR / "models" / "deploy.prototxt"
MODEL_PATH = BASE_DIR / "models" / "mobilenet_iter_73000.caffemodel"


# ============================================================
# MobileNet-SSD classes
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


CONFIDENCE_THRESHOLD = 0.5


# ============================================================
# Load model
# ============================================================

def load_model():

    net = cv2.dnn.readNetFromCaffe(
        str(PROTOTXT_PATH),
        str(MODEL_PATH)
    )

    print("MobileNet-SSD model loaded successfully!")

    return net


# ============================================================
# Detect objects
# ============================================================

def detect_objects(image, net):

    # Convert image into a blob
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False
    )

    # Give blob to the neural network
    net.setInput(blob)

    # Run inference
    detections = net.forward()

    results = []

    # Image dimensions
    (h, w) = image.shape[:2]

    # Process detections
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Ignore low-confidence detections
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Get class ID
        class_id = int(detections[0, 0, i, 1])

        # Convert class ID to label
        label = CLASSES[class_id]

        # Get normalized bounding box
        box = detections[0, 0, i, 3:7]

        # Convert normalized coordinates to pixels
        startX = int(box[0] * w)
        startY = int(box[1] * h)

        endX = int(box[2] * w)
        endY = int(box[3] * h)

        results.append({
            "label": label,
            "confidence": float(confidence),
            "box": (startX, startY, endX, endY)
        })

    return results


# ============================================================
# Draw detections
# ============================================================

def draw_detections(image, results):

    output = image.copy()

    for result in results:

        label = result["label"]
        confidence = result["confidence"]

        startX, startY, endX, endY = result["box"]

        # -----------------------------------------
        # Draw bounding box
        # -----------------------------------------

        cv2.rectangle(
            output,
            (startX, startY),
            (endX, endY),
            (0, 255, 0),
            2
        )

        # -----------------------------------------
        # Create label
        # -----------------------------------------

        text = f"{label.title()} {confidence * 100:.2f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        # -----------------------------------------
        # Label background coordinates
        # -----------------------------------------

        label_top = max(
            startY - text_height - baseline - 5,
            0
        )

        label_bottom = startY

        label_right = startX + text_width + 10

        # Draw filled rectangle behind text
        cv2.rectangle(
            output,
            (startX, label_top),
            (label_right, label_bottom),
            (0, 255, 0),
            -1
        )

        # -----------------------------------------
        # Draw label text
        # -----------------------------------------

        cv2.putText(
            output,
            text,
            (startX + 5, label_bottom - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    return output
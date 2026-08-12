import cv2

from src.detector import (
    load_model,
    detect_objects,
    draw_detections
)


# ============================================================
# Load model
# ============================================================

net = load_model()


# ============================================================
# Load image
# ============================================================

image_path = "images/cat.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

print("Image loaded successfully!")
print("Image shape:", image.shape)


# ============================================================
# Detect objects
# ============================================================

results = detect_objects(
    image,
    net
)


# ============================================================
# Display detection results
# ============================================================

for result in results:

    print(
        f"Detected: {result['label']} "
        f"| Confidence: {result['confidence']:.2f} "
        f"| Box: {result['box']}"
    )


# ============================================================
# Draw detections
# ============================================================

output = draw_detections(
    image,
    results
)


# ============================================================
# Save output
# ============================================================

output_path = "images/output.jpg"

cv2.imwrite(
    output_path,
    output
)

print(f"Output saved to: {output_path}")
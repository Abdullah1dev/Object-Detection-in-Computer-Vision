from src.detector import net, image, blob , detections

print("Model loaded:")
print(net)

print("Image shape:", image.shape)

print("Blob shape:", blob.shape)


print("Detection shape:", detections.shape)
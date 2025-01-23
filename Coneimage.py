from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('conemodel1.pt')

# Load the image
image_path = r"C:\Users\shrim\OneDrive\Desktop\Hackwar\use-of-traffic-cones-768x512.jpg"
image = cv2.imread(image_path)

# Run prediction on the image
results = model(image, conf=0.25)

# Render results on the image
annotated_image = results[0].plot()

# Resize the image to the desired frame size (for example, 800x600)
resized_image = cv2.resize(annotated_image, (800, 600))

# Display the resized image with predictions
cv2.imshow('YOLO Prediction on Image', resized_image)

print(results[0])

# Wait until a key is pressed and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

# Load the YOLO model
model =  YOLO('conemodel1.pt')

# Open the webcam
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the frame
    results = model(frame, conf=0.25)

    # Render results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with predictions
    cv2.imshow('YOLO Prediction', annotated_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
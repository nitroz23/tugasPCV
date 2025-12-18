import cv2
import numpy as np

# HSV color range for BLUE
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

# Morphology kernel
KERNEL = np.ones((5, 5), np.uint8)
MIN_AREA = 1500

# Start webcam
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_cam = cv2.VideoWriter('cam_output.mp4', fourcc, 20.0, (frame_width, frame_height))
out_mask = cv2.VideoWriter('mask_output.mp4', fourcc, 20.0, (frame_width, frame_height), False)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold blue color
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # Noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    message = "No blue object detected"
    color = (0, 0, 255)

    if contours:
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            message = "Blue Object Detected!"
            color = (255, 0, 0)

    # Display status text
    cv2.putText(frame, message, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Show results
    cv2.imshow("Webcam", frame)
    cv2.imshow("Blue Mask", mask)
    out_cam.write(frame)
    out_mask.write(mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
out_cam.release()
out_mask.release()
cv2.destroyAllWindows()

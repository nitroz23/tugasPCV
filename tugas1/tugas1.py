import cv2
import numpy as np

def median(image):
    return cv2.medianBlur(image, 5)

def gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera error")
        return

    # Effect counters (STATE)
    effects = {
        "median": 0,
        "gaussian": 0,
        "sharpen": 0
    }

    # Get the default frame width and height
    # frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        processed = frame.copy()

        # Apply effects according to counters
        for _ in range(effects["median"]):
            processed = median(processed)

        for _ in range(effects["gaussian"]):
            processed = gaussian(processed)

        for _ in range(effects["sharpen"]):
            processed = sharpen(processed)

        # Overlay text
        y = 30
        for name, count in effects.items():
            text = f"{name.capitalize()}: {count}"
            cv2.putText(
                processed,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y += 30
        
        # out.write(processed)

        cv2.imshow("Camera", processed)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('z'):
            effects["median"] += 1
        elif key == ord('x'):
            effects["gaussian"] += 1
        elif key == ord('c'):
            effects["sharpen"] += 1
        elif key == ord('m'):
            for k in effects:
                effects[k] = 0
        elif key == ord('q'):
            break

    cam.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

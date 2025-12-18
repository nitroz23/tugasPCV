import cv2
import numpy as np

def apply_median_blur(image, kernel_size=5):
    median_blur = cv2.medianBlur(image, kernel_size)
    return median_blur

def apply_gaussian_blur(image, kernel_size=5):
    gaussian_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return gaussian_blur

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def main():
    path = 'assets/tmoperao.jpg'
    image = cv2.imread(path)
    original_image = image.copy()

    window_name = 'Image Processing Test'

    while True:
        cv2.imshow(window_name, image)

        key = cv2.waitKey(0) & 0xFF  # wait indefinitely for key press

        if key == ord('z'):
            image = apply_median_blur(image)
        elif key == ord('x'):
            image = apply_gaussian_blur(image)
        elif key == ord('c'):
            image = apply_sharpening(image)
        elif key == ord('m'):
            image = original_image.copy()
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
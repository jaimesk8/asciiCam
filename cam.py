import cv2
import numpy as np
import os

# ASCII characters set from dark to light
ASCII_CHARS = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]

# Function to map pixel intensity to an ASCII character
def pixel_to_ascii(pixel_value):
    return ASCII_CHARS[int(pixel_value // 50)]  # 255/10 = 25, 10 is the length of ASCII_CHARS

# Function to convert image to grayscale
def grayify(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to resize the image
def resize_image(image, new_width=200):
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Function to convert image to ASCII
def image_to_ascii(image):
    grayscale_image = grayify(image)
    ascii_str = ""
    for pixel_value in grayscale_image.flatten():
        ascii_str += pixel_to_ascii(pixel_value)
    return ascii_str

# Function to convert ASCII to image for display in OpenCV window
def ascii_to_image(ascii_str, width=200):
    # Calculate the height based on the ASCII string length and width
    ascii_lines = [ascii_str[i:i+width] for i in range(0, len(ascii_str), width)]
    height = len(ascii_lines)

    # Create a blank white image
    img = np.ones((height * 15, width * 10, 3), dtype=np.uint8) * 255

    # Write ASCII characters onto the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    for y, line in enumerate(ascii_lines):
        for x, char in enumerate(line):
            # Put each character on the image
            cv2.putText(img, char, (x * 10, (y + 1) * 15), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img

# Function to show the live video feed with the ASCII filter applied
def show_video_with_ascii_filter():
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize the frame to fit in the ASCII window
        resized_frame = resize_image(frame, new_width=200)
        
        # Convert the resized frame to ASCII art
        ascii_str = image_to_ascii(resized_frame)
        
        # Convert the ASCII string back to an image for displaying in OpenCV window
        ascii_image = ascii_to_image(ascii_str, width=200)
        
        # Show the ASCII art as a filter in the OpenCV window
        cv2.imshow('ASCII Filtered Camera Feed', ascii_image)
        
        # Check if the 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_video_with_ascii_filter()

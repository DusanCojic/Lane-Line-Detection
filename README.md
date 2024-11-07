### Lane Line Detection using OpenCV

This project demonstrates the detection of lane lines in road images or video streams using OpenCV in Python. It applies computer vision techniques to identify the lane markings, which are essential for self-driving car systems and advanced driver-assistance systems (ADAS).

#### Features
- Lane detection on static images or video streams.
- Use of edge detection and region-of-interest (ROI) masking to focus on lanes.
- Hough Transform to detect straight lines.

#### Code Walkthrough
1. Preprocessing:
Grayscale Conversion: Convert the image to grayscale to simplify the analysis.
Gaussian Blur: Apply a Gaussian blur to smooth the image and reduce noise.
Canny Edge Detection: Detect edges in the image using the Canny edge detector.
2. Region of Interest (ROI):
A polygonal region is defined to focus the detection on the relevant areas (lane lines) and ignore other parts of the image (e.g., sky, non-road areas).
3. Line Detection:
Hough Transform: Used to detect straight lines in the image.
Lines are filtered based on their slope and length to ensure they correspond to the lanes.
4. Lane Line Overlay:
Detected lanes are drawn onto the original image for visualization.

https://github.com/user-attachments/assets/3bcc7a96-68f0-4d2b-8d8c-9aff73db660c

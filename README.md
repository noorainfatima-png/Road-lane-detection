
#  Road Lane Detection using Python & Computer Vision

An automated computer vision pipeline designed to detect and track highway driving lanes in real time from video feeds. Using fundamental image processing techniques—such as HSV color space filtering, Canny edge detection, and Hough Line transforms—this system calculates lane trajectories to provide a foundational setup for Advanced Driver Assistance Systems (ADAS).

---

## Tech Stack & Prerequisites

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Numeric Computing:** NumPy
* **Data Visualization:** Matplotlib

---

##  Computer Vision Pipeline


[ Raw Video Input ] ──► [ HSV & Grayscale Mask ] ──► [ Gaussian Blur Noise Reduction ]
                                                                 │
[ Rendered Video Output ] ◄── [ Alpha Overlay ] ◄── [ Polyfit Extrapolation ] ◄── [ Canny & Hough Lines ]



## 📁 Repository Structure

```
road-lane-detection/
│
├── input/
│   └── road.mp4             # Raw input video feed
│
├── output/
│   └── lane_detected.mp4    # Rendered output video with lane overlays
│
├── images/                  # Static visual documentation assets
│   ├── input.png
│   └── output.png
│
├── lane_detection.py        # Core processing pipeline script
├── requirements.txt         # Project dependencies
├── .gitignore               # System ignore rules
└── README.md                # Project documentation

```
## The lane detection process includes:

1. Reading the input road video
2. Converting frames to grayscale
3. Applying Gaussian Blur
4. Detecting edges using Canny Edge Detection
5. Defining the Region of Interest
6. Detecting lane lines using Hough Line Transform
7. Drawing the detected lane lines
8. Generating the processed result


##  Roadmap & Future Enhancements

* [ ] **Temporal Moving Average Filter:** Implement a frame-buffer pipeline to average slope vectors across consecutive frames and eliminate high-frequency visual jitter.
* [ ] **Curved Lane Tracking:** Upgrade from linear ($1^\text{st}$-order) polynomial fitting to quadratic ($2^\text{nd}$-order) curves to handle sharp turns accurately.
* [ ] **Deep Learning Integration:** Implement lightweight semantic segmentation (e.g., U-Net or Ultra-Fast-Lane-Detection) for adverse weather and unpainted roads.



## 📁 Project Structure

```text
road-lane-detection/
│
├── input/
│   └── input.mp4
│
├── images/
│   ├── input.png
│   └── output.png
│
├── lane_detection.py
├── requirements.txt
├── README.md
└── .gitignore

▶️ How to Run

Install the required libraries:

pip install -r requirements.txt

▶️ Run the program:

python lane_detection.py

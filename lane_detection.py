
import cv2
import numpy as np
import os


# ============================================================
# 1. DETECT LANE LINES IN ONE FRAME
# ============================================================

def detect_lane_lines(frame):

    # --------------------------------------------------------
    # Convert the frame to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # --------------------------------------------------------
    # Detect white lane markings
    # --------------------------------------------------------

    white_mask = cv2.inRange(
        gray,
        180,
        255
    )


    # --------------------------------------------------------
    # Detect yellow lane markings
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    lower_yellow = np.array([
        15,
        80,
        80
    ])

    upper_yellow = np.array([
        40,
        255,
        255
    ])

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )


    # --------------------------------------------------------
    # Combine white and yellow lane masks
    # --------------------------------------------------------

    lane_mask = cv2.bitwise_or(
        white_mask,
        yellow_mask
    )


    # Apply the mask to the grayscale image
    masked = cv2.bitwise_and(
        gray,
        lane_mask
    )


    # --------------------------------------------------------
    # Apply Gaussian Blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        masked,
        (5, 5),
        0
    )


    # --------------------------------------------------------
    # Detect edges using Canny
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )


    # ========================================================
    # 2. REGION OF INTEREST
    # ========================================================

    height, width = frame.shape[:2]


    # Define the road area
    roi_vertices = np.array([
        [
            (int(width * 0.10), height),
            (int(width * 0.45), int(height * 0.60)),
            (int(width * 0.55), int(height * 0.60)),
            (int(width * 0.90), height)
        ]
    ], dtype=np.int32)


    # Create a black mask
    mask = np.zeros_like(edges)


    # Fill the road-shaped region with white
    cv2.fillPoly(
        mask,
        roi_vertices,
        255
    )


    # Keep only the region of interest
    roi = cv2.bitwise_and(
        edges,
        mask
    )


    # ========================================================
    # 3. HOUGH LINE TRANSFORM
    # ========================================================

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )


    # Create an empty image for drawing lane lines
    line_image = np.zeros_like(frame)


    # Lists for left and right lane lines
    left_lines = []
    right_lines = []


    # --------------------------------------------------------
    # Separate detected lines into left and right lanes
    # --------------------------------------------------------

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line


            # Avoid division by zero
            if x2 == x1:
                continue


            # Calculate slope
            slope = (y2 - y1) / (x2 - x1)


            # Ignore nearly horizontal lines
            if abs(slope) < 0.5:
                continue


            # Negative slope = left lane
            if slope < 0:

                left_lines.append(
                    (x1, y1, x2, y2)
                )


            # Positive slope = right lane
            else:

                right_lines.append(
                    (x1, y1, x2, y2)
                )


    # ========================================================
    # 4. DRAW AVERAGED LANE LINES
    # ========================================================

    def draw_average_line(lines):

        if len(lines) == 0:
            return


        # Store all x and y coordinates
        x_values = []
        y_values = []


        for x1, y1, x2, y2 in lines:

            x_values.extend([
                x1,
                x2
            ])

            y_values.extend([
                y1,
                y2
            ])


        # Fit a straight line through the points
        slope, intercept = np.polyfit(
            x_values,
            y_values,
            1
        )


        # Prevent division by zero
        if slope == 0:
            return


        # Bottom of the frame
        y_bottom = height


        # Upper point of the lane
        y_top = int(
            height * 0.60
        )


        # Calculate x coordinates
        x_bottom = int(
            (y_bottom - intercept) / slope
        )

        x_top = int(
            (y_top - intercept) / slope
        )


        # Draw the lane line
        cv2.line(
            line_image,
            (x_bottom, y_bottom),
            (x_top, y_top),
            (0, 0, 255),
            8
        )


    # Draw left lane
    draw_average_line(left_lines)


    # Draw right lane
    draw_average_line(right_lines)


    # ========================================================
    # 5. COMBINE ORIGINAL FRAME + LANE LINES
    # ========================================================

    result = cv2.addWeighted(
        frame,
        0.8,
        line_image,
        1,
        0
    )


    return result


# ============================================================
# 6. INPUT AND OUTPUT VIDEO PATHS
# ============================================================

input_video = "input/road.mp4"

output_video = "output/lane_detected.mp4"


# ============================================================
# 7. CHECK WHETHER INPUT VIDEO EXISTS
# ============================================================

if not os.path.exists(input_video):

    print()
    print("ERROR: Input video was not found.")
    print()
    print("Expected location:")
    print(input_video)
    print()

    exit()


# ============================================================
# 8. OPEN THE INPUT VIDEO
# ============================================================

cap = cv2.VideoCapture(
    input_video
)


if not cap.isOpened():

    print()
    print("ERROR: Could not open the input video.")
    print()

    exit()


# ============================================================
# 9. GET VIDEO INFORMATION
# ============================================================

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)


# ============================================================
# 10. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "output",
    exist_ok=True
)


# ============================================================
# 11. CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 12. PROCESS THE VIDEO
# ============================================================

print()
print("========================================")
print("     ROAD LANE LINE DETECTION")
print("========================================")
print()
print("Input video:")
print(input_video)
print()
print("Processing video...")
print()


frame_count = 0


while True:

    # Read one frame
    ret, frame = cap.read()


    # Stop when there are no more frames
    if not ret:
        break


    # Detect lane lines
    result = detect_lane_lines(
        frame
    )


    # Write processed frame to output video
    out.write(
        result
    )


    frame_count += 1


    # Show progress every 50 frames
    if frame_count % 50 == 0:

        print(
            f"Processed {frame_count} frames..."
        )


# ============================================================
# 13. RELEASE VIDEO RESOURCES
# ============================================================

cap.release()

out.release()

cv2.destroyAllWindows()


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print()
print("========================================")
print("          PROCESSING COMPLETE")
print("========================================")
print()
print(f"Total frames processed: {frame_count}")
print()
print("Output video:")
print(output_video)
print()
print("Open the output folder to view the result.")
print()


import cv2
import numpy as np
from collections import deque
from config import OUTPUT_DIR

H = np.load(f"{OUTPUT_DIR}/homography.npy")
OUTPUT_W, OUTPUT_H = 600, 400

# obtain capture resolution
cap_test = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_00.mp4")
clip_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
clip_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))

# upload the background ground truth (empty table)
background = cv2.imread(f"{OUTPUT_DIR}/screenshot_00.png")

# Give to the background the same resolution of the capture
background_resized = cv2.resize(background, (clip_w, clip_h))

background_warped = cv2.warpPerspective(background, H, (OUTPUT_W, OUTPUT_H))
background_gray = cv2.cvtColor(background_warped, cv2.COLOR_BGR2GRAY)
#cv2.imshow("background3", background_gray)
#cv2.waitKey(0)

# Parameters for HoughCircles
BLUR_KERNEL = 7
DP           = 1.3 # accumulator resolution
MIN_DIST     = 20
PARAM1       = 50 # Canny threshold
PARAM2       = 18 # accumulator threshold
MIN_RADIUS   = 13
MAX_RADIUS   = 20

# Define a margin, to avoid to recognize the holes as balls 
MARGIN = 35  # pixels

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask,
              (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN),
              255, -1)


# threshold for binarization
THRESHOLD = 20

# kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

# Frames for averaging
N_FRAMES = 20
frame_buffer = deque(maxlen=N_FRAMES)

# save clip
# Output parameters
"""
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = capture.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter("PROJECT/clips/balls_detection_01.mp4", fourcc, fps, (OUTPUT_W, OUTPUT_H))
OUTPUT_DIR = "PROJECT/clips"
"""

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_04.mp4")
#capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_homography = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
    gray = cv2.cvtColor(frame_homography, cv2.COLOR_BGR2GRAY)

    # substraction (frame - background)
    diff = cv2.absdiff(gray, background_gray)
    #diff = cv2.GaussianBlur(diff, (BLUR_KERNEL, BLUR_KERNEL), 0)
    _, binary = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    # morphological operations for cleaning (try with ellipse kernel as well)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    binary = cv2.bitwise_and(binary, binary, mask=margin_mask)

    # average of last frames
    frame_buffer.append(binary)
    if len(frame_buffer) < N_FRAMES:
        """Wait to get N_FRAMES for initialization"""
        cv2.waitKey(30)
        continue

    averaged = np.mean(frame_buffer, axis=0).astype(np.uint8)
    _, averaged_binary = cv2.threshold(averaged, 100, 255, cv2.THRESH_BINARY)


    circles = cv2.HoughCircles(
        averaged_binary,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=PARAM1,
        param2=PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    output = frame_homography.copy()

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(output, f"Balls: {len(circles)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(output, "No balls found!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Original", frame_homography)
    cv2.imshow("Difference", diff)
    cv2.imshow("Binary mask", binary)
    cv2.imshow("Temporal avg", averaged_binary)
    cv2.imshow("Balls detection", output)

    # save clip
    #writer.write(output)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        """ print the parameters of the circles, save frames for show results """
        #for circle in circles:
        #    print(circle)
        break
    if key == ord('s'):
        """Save screenshots of results"""
        cv2.imwrite("PROJECT/RESULTS/v02_input.png", frame_homography)
        #cv2.imwrite("PROJECT/images/balls_detection_02.png", blurred)
        cv2.imwrite("PROJECT/images/v02_output.png", output)
        #cv2.imwrite("PROJECT/images/balls_detection_04.png", roi_mask)


capture.release()
#writer.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import os
from config import OUTPUT_DIR

#capture = cv2.VideoCapture("PROJECT/clips/clip_02.mp4")  # 1st experiment
#capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_00.mp4")

#H = np.load("PROJECT/homography.npy") # 1st experiment
H = np.load(f"{OUTPUT_DIR}/homography.npy")
OUTPUT_W, OUTPUT_H = 600, 400

# Parameters for HoughCircles
BLUR_KERNEL  = 9 
DP           = 1.3 # accumulator resolution
MIN_DIST     = 20
PARAM1       = 50 # Canny threshold
PARAM2       = 18 # accumulator threshold
MIN_RADIUS   = 11
MAX_RADIUS   = 20

# Define a margin, to avoid to recognize the holes as balls 
MARGIN = 50  # pixels

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask,
              (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN),
              255, -1)


# save clip
"""
# Output parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = capture.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter("PROJECT/clips/balls_detection_01.mp4", fourcc, fps, (OUTPUT_W, OUTPUT_H))
OUTPUT_DIR = "PROJECT/clips"
"""
while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_homography = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
    gray = cv2.cvtColor(frame_homography, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

    blurred_mask = cv2.bitwise_and(blurred, blurred, mask=margin_mask)

    circles = cv2.HoughCircles(
        blurred_mask,
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
    cv2.imshow("Gray + Blur", blurred)
    cv2.imshow("Balls detection", output)

    # save clip
    #writer.write(output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        """ print the parameters of the circles, save frames for show results """
        #for circle in circles:
        #    print(circle)
        # cv2.imwrite("PROJECT/images/v01_balls_detection_01.png", frame_homography)
        # cv2.imwrite("PROJECT/images/v01_balls_detection_02.png", blurred)
        # cv2.imwrite("PROJECT/images/v01_balls_detection_03.png", output)
        # cv2.imwrite("PROJECT/images/v01_balls_detection_04.png", roi_mask)
        break

capture.release()
#writer.release()
cv2.destroyAllWindows()
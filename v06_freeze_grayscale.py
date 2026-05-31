import cv2
import numpy as np
from collections import deque
from config import OUTPUT_DIR

H = np.load(f"{OUTPUT_DIR}/homography.npy")
OUTPUT_W, OUTPUT_H = 600, 400

# obtain capture resolution
cap_test = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_00.mp4")
#cap_test = cv2.VideoCapture("PROJECT/clips/clip_02.mp4")

clip_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
clip_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_test.release()

# upload the background ground truth (empty table)
background = cv2.imread(f"{OUTPUT_DIR}/screenshot_00.png")
#background = cv2.imread("PROJECT/images/capture_frame_02.png")

background_resized = cv2.resize(background, (clip_w, clip_h))
background_warped = cv2.warpPerspective(background_resized, H, (OUTPUT_W, OUTPUT_H))
background_gray = cv2.cvtColor(background_warped, cv2.COLOR_BGR2GRAY)
#cv2.imshow("background3", background_warped)
#cv2.waitKey(0)

# Parameters for HoughCircles, thresholding and margin mask
BLUR_KERNEL = 9
DP           = 1.3 # accumulator resolution
PARAM1       = 50 # Canny threshold
PARAM2       = 15 # accumulator threshold
MIN_RADIUS   = 13
MIN_DIST     = MIN_RADIUS * 1.45
MAX_RADIUS   = 17

MARGIN = 45  # pixels
THRESHOLD = 50

# kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN), 255, -1)

# initialization
detected_circles = None
acquiring = False
acq_buffer = []
ACQ_FRAMES = 30
diff = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
binary = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

# save clip
# Output parameters
"""
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = capture.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter("PROJECT/clips/balls_detection_01.mp4", fourcc, fps, (OUTPUT_W, OUTPUT_H))
OUTPUT_DIR = "PROJECT/clips"
"""
img_count = 0

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_04.mp4")
#capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#capture = cv2.VideoCapture("PROJECT/clips/clip_02.mp4")

fps = 60
delay = int(1000 / fps)  # ms da aspettare per frame



while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_homography = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))

    key = cv2.waitKey(delay) & 0xFF 

    if key == ord(' ') and not acquiring:
        """Press SpaceBar for get a series of n frames"""
        acquiring = True
        acq_buffer = []

    if acquiring:
        freeze = frame_homography.copy()
        gray = cv2.cvtColor(freeze, cv2.COLOR_BGR2GRAY)

        # substraction (frame - background)
        diff = cv2.absdiff(gray, background_gray)
        acq_buffer.append(gray.astype(np.float32))
        cv2.putText(frame_homography, f"Acquiring... {len(acq_buffer)}/{ACQ_FRAMES}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2) 

        if len(acq_buffer) == ACQ_FRAMES:
            acquiring = False

            avg_diff = np.mean(acq_buffer, axis=0).astype(np.uint8)
            avg_diff = cv2.GaussianBlur(avg_diff, (BLUR_KERNEL, BLUR_KERNEL), 0)
            
            #_, binary = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

            # morphological operations for cleaning (try with ellipse kernel as well)
            #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel1)
            #binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE,  kernel2)

            # apply mask for avoid margins
            binary = cv2.bitwise_and(avg_diff, avg_diff, mask=margin_mask)


            circles = cv2.HoughCircles(
                binary,
                cv2.HOUGH_GRADIENT,
                dp=DP,
                minDist=MIN_DIST,
                param1=PARAM1,
                param2=PARAM2,
                minRadius=MIN_RADIUS,
                maxRadius=MAX_RADIUS
            )

            if circles is not None:
                detected_circles = [(int(x), int(y), int(r)) for x, y, r in np.round(circles[0])]
            else: 
                detected_circles = []
    
    # draw the circles on the recording / live feed
    if detected_circles:
        for x,y,r in detected_circles:
            cv2.circle(frame_homography, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame_homography, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(frame_homography, f"Balls: {len(detected_circles)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif detected_circles is not None:
        cv2.putText(frame_homography, "No balls found!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else: 
        cv2.putText(frame_homography, "Press SPACEBAR to find balls before to shoot", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
    
    cv2.imshow("Difference", binary)
    #cv2.imshow("Binary mask", binary)
    cv2.imshow("Balls detection", frame_homography)

    # save clip
    #writer.write(output)

    if key == ord('q'):
        break
    if key == ord('s'):
        """ print the parameters of the circles, save frames for show results """
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v06_resultBinary_0{img_count}.png", binary)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v06_resultOutput_0{img_count}.png", frame_homography)
        img_count+=1
        for circle in detected_circles:
            print(circle)


capture.release()
#writer.release()
cv2.destroyAllWindows()
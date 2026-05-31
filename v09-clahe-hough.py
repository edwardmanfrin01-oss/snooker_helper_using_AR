import cv2
import numpy as np
from config import OUTPUT_DIR

H = np.load(f"{OUTPUT_DIR}/homography.npy")
OUTPUT_W, OUTPUT_H = 600, 400

cap_test = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_00.mp4")
clip_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
clip_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_test.release()

background = cv2.imread(f"{OUTPUT_DIR}/screenshot_00.png")
background_resized = cv2.resize(background, (clip_w, clip_h))
background_warped  = cv2.warpPerspective(background_resized, H, (OUTPUT_W, OUTPUT_H))
background_gray    = cv2.cvtColor(background_warped, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------------------
# Parameters

BLUR_KERNEL = 3    # wider blur softens shadow edges and stripe details before Hough
DP          = 1.3
PARAM1      = 50
PARAM2      = 17 
MIN_RADIUS  = 14
MAX_RADIUS  = 17
MIN_DIST    = MIN_RADIUS * 1.6 
MARGIN      = 47
ACQ_FRAMES  = 15 

CLAHE_CLIP  = 1.5
CLAHE_TILE  = (6, 6)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN), 255, -1)

# ---------------------------------------------------------------------------
# State

detected_circles = None
acquiring        = False
acq_buffer       = []
img_count        = 0

display_gray  = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
display_clahe = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_03.mp4")
#capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

fps   = 60
delay = int(1000 / fps)

# ---------------------------------------------------------------------------
# Main loop

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_h = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
    key = cv2.waitKey(delay) & 0xFF

    if key == ord(' ') and not acquiring:
        acquiring   = True
        acq_buffer  = []

    if acquiring:
        gray = cv2.cvtColor(frame_h, cv2.COLOR_BGR2GRAY)
        acq_buffer.append(gray.astype(np.float32))
        cv2.putText(frame_h, f"Acquiring... {len(acq_buffer)}/{ACQ_FRAMES}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if len(acq_buffer) == ACQ_FRAMES:
            acquiring = False

            # 1. Average frames
            avg_gray = np.mean(acq_buffer, axis=0).astype(np.uint8)

            # 2. Blur
            avg_blur = cv2.GaussianBlur(avg_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

            # 3. CLAHE
            avg_clahe = clahe.apply(avg_blur)

            # 4. Margin mask + HoughCircles on CLAHE
            masked = cv2.bitwise_and(avg_clahe, avg_clahe, mask=margin_mask)
            circles = cv2.HoughCircles(
                masked, cv2.HOUGH_GRADIENT,
                dp=DP, minDist=MIN_DIST,
                param1=PARAM1, param2=PARAM2,
                minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
            )

            if circles is not None:
                detected_circles = [(int(x), int(y), int(r))
                                    for x, y, r in np.round(circles[0])]
            else:
                detected_circles = []

            # keep for debug windows
            display_gray  = cv2.bitwise_and(avg_blur,  avg_blur,  mask=margin_mask)
            display_clahe = masked

    # Draw
    if detected_circles:
        for x, y, r in detected_circles:
            cv2.circle(frame_h, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame_h, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(frame_h, f"Balls: {len(detected_circles)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif detected_circles is not None:
        cv2.putText(frame_h, "No balls found!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame_h, "Press SPACEBAR to detect balls",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    cv2.imshow("Gray (no CLAHE)", display_gray)
    cv2.imshow("Gray (CLAHE)", display_clahe)
    cv2.imshow("Balls detection", frame_h)

    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v09_resultGray_0{img_count}.png", display_gray)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v09_resultCLAHE_0{img_count}.png", display_clahe)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v09_resultOutput_0{img_count}.png", frame_h)
        img_count += 1
        print(f"Saved {img_count}  Balls: {len(detected_circles)}")
        for c in detected_circles:
            print(c)

capture.release()
cv2.destroyAllWindows()

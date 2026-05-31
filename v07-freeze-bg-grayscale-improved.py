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
background_warped = cv2.warpPerspective(background_resized, H, (OUTPUT_W, OUTPUT_H))
background_gray = cv2.cvtColor(background_warped, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------------------
# Parameters

BLUR_KERNEL = 3
DP          = 1.3
PARAM1      = 50
PARAM2_GRAY = 17
PARAM2_DIFF = 12
MIN_RADIUS  = 14
MAX_RADIUS  = 17
MIN_DIST_1  = MIN_RADIUS * 1.8
MIN_DIST_2  = MIN_RADIUS * 1.8
MARGIN      = 42
ACQ_FRAMES  = 15
MERGE_DIST  = MIN_RADIUS * 2

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN), 255, -1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_hough(img, param2, min_dist):
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT,
        dp=DP, minDist=min_dist,
        param1=PARAM1, param2=param2,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
    )
    if circles is not None:
        return [(int(x), int(y), int(r)) for x, y, r in np.round(circles[0])]
    return []


def merge_circles(primary, secondary, dist_threshold):
    """Append circles from secondary that don't overlap with any in primary."""
    merged = list(primary)
    added = []
    for cx, cy, cr in secondary:
        if all(np.hypot(cx - mx, cy - my) >= dist_threshold for mx, my, _ in merged):
            merged.append((cx, cy, cr))
            added.append((cx, cy, cr))
    return merged, added


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
detected_circles = None   # None = not yet acquired; [] = acquired but empty
circles_from_gray = []    # for color-coded display
circles_from_diff = []
acquiring = False
acq_buffer_gray = []
acq_buffer_diff = []
img_count = 0

binary_gray = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
binary_diff = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_06.mp4")
#capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

fps   = 60
delay = int(1000 / fps)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_homography = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
    key = cv2.waitKey(delay) & 0xFF

    if key == ord(' ') and not acquiring:
        acquiring = True
        acq_buffer_gray = []
        acq_buffer_diff = []

    if acquiring:
        gray = cv2.cvtColor(frame_homography, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background_gray)
        acq_buffer_gray.append(gray.astype(np.float32))
        acq_buffer_diff.append(diff.astype(np.float32))
        cv2.putText(frame_homography, f"Acquiring... {len(acq_buffer_gray)}/{ACQ_FRAMES}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if len(acq_buffer_gray) == ACQ_FRAMES:
            acquiring = False

            # --- Pass 1: raw gray average (strong edges on all balls) ---
            avg_gray = np.mean(acq_buffer_gray, axis=0).astype(np.uint8)
            avg_gray = cv2.GaussianBlur(avg_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
            binary_gray = cv2.bitwise_and(avg_gray, avg_gray, mask=margin_mask)
            circles_from_gray = run_hough(binary_gray, PARAM2_GRAY, MIN_DIST_1)

            # --- Pass 2: background subtraction (sensitive to low-contrast balls) ---
            avg_diff = np.mean(acq_buffer_diff, axis=0).astype(np.uint8)
            avg_diff = cv2.GaussianBlur(avg_diff, (BLUR_KERNEL, BLUR_KERNEL), 0)
            # close fills gaps within blobs; open removes thin shadow artifacts
            avg_diff = cv2.morphologyEx(avg_diff, cv2.MORPH_CLOSE, kernel_close)
            avg_diff = cv2.morphologyEx(avg_diff, cv2.MORPH_OPEN,  kernel_open)
            binary_diff = cv2.bitwise_and(avg_diff, avg_diff, mask=margin_mask)
            circles_from_diff_raw = run_hough(binary_diff, PARAM2_DIFF, MIN_DIST_2)

            # --- Merge: keep pass-1 results, add pass-2 detections not yet covered ---
            detected_circles, circles_from_diff = merge_circles(
                circles_from_gray, circles_from_diff_raw, MERGE_DIST
            )

    # --- Draw ---
    if detected_circles:
        # Green = detected by pass 1 (raw gray)
        for x, y, r in circles_from_gray:
            cv2.circle(frame_homography, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame_homography, (x, y), 3, (0, 255, 0), -1)
        # Cyan = added by pass 2 (bg subtraction) — likely the green ball or shadows
        for x, y, r in circles_from_diff:
            cv2.circle(frame_homography, (x, y), r, (0, 0, 255), 2)
            cv2.circle(frame_homography, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(frame_homography,
                    f"Balls: {len(detected_circles)}  (gray:{len(circles_from_gray)}  diff:+{len(circles_from_diff)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif detected_circles is not None:
        cv2.putText(frame_homography, "No balls found!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame_homography, "Press SPACEBAR to detect balls",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    cv2.imshow("Pass 1 - Gray avg", binary_gray)
    cv2.imshow("Pass 2 - Diff avg", binary_diff)
    cv2.imshow("Balls detection", frame_homography)

    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v07_resultBinary_gray_0{img_count}.png", binary_gray)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v07_resultBinary_diff_0{img_count}.png", binary_diff)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v07_resultOutput_0{img_count}.png", frame_homography)
        img_count += 1
        print(f"Saved {img_count}")
        print(f"Pass 1 (gray): {len(circles_from_gray)} circles")
        print(f"Pass 2 (diff): +{len(circles_from_diff)} new circles")
        for c in detected_circles:
            print(c)

capture.release()
cv2.destroyAllWindows()

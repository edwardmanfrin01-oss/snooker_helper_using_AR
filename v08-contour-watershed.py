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

BLUR_KERNEL        = 9
THRESHOLD          = 45     # raised: shadows have diff 15-30, balls 40+; avoids blob merging
MIN_RADIUS         = 13
MAX_RADIUS         = 17
CIRCULARITY_THRESH = 0.6   # 1.0 = perfect circle; shadows/noise score much lower
ASPECT_MIN         = 0.70   # bounding rect width/height ratio bounds
ASPECT_MAX         = 1.30
WS_DIST_THRESH     = 0.35   # slightly lower: separates touching balls more aggressively
MARGIN             = 45
ACQ_FRAMES         = 30

BALL_AREA_MIN = np.pi * MIN_RADIUS**2 * 0.5
BALL_AREA_MAX = np.pi * MAX_RADIUS**2 * 2.0

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # smaller: avoids bridging nearby blobs
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_dil   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN), 255, -1)

# ---------------------------------------------------------------------------
# Detection pipeline


def build_binary(avg_gray):
    """Background subtraction → blur → threshold → morphology → mask."""
    diff = cv2.absdiff(avg_gray, background_gray)
    diff = cv2.GaussianBlur(diff, (BLUR_KERNEL, BLUR_KERNEL), 0)
    _, binary = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    binary = cv2.bitwise_and(binary, binary, mask=margin_mask)
    return binary


def watershed_markers(binary, avg_bgr):
    """
    Distance transform → local maxima as seeds → watershed.
    Returns the labelled marker image (-1 = boundary, 1 = bg, >1 = regions).
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return None, dist

    _, sure_fg = cv2.threshold(dist, WS_DIST_THRESH * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, kernel_dil, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 
    markers[unknown == 255] = 0 

    cv2.watershed(avg_bgr, markers)
    return markers, dist


def filter_regions(markers):
    """
    For each watershed region check area, circularity, aspect ratio.
    Returns list of (cx, cy, r) for valid balls.
    """
    circles = []
    for label in np.unique(markers):
        if label <= 1 or label == -1:   # skip background and watershed borders
            continue

        region = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
        region[markers == label] = 255

        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        if not (BALL_AREA_MIN <= area <= BALL_AREA_MAX):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < CIRCULARITY_THRESH:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        if not (ASPECT_MIN <= float(w) / h <= ASPECT_MAX):
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circles.append((int(cx), int(cy), int(r)))

    return circles


def detect_balls(avg_bgr, avg_gray):
    binary = build_binary(avg_gray)
    markers, dist = watershed_markers(binary, avg_bgr)

    if markers is None:
        return [], binary, np.zeros_like(binary)

    circles = filter_regions(markers)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return circles, binary, dist_vis


# ---------------------------------------------------------------------------
# State

detected_circles = None  
acquiring        = False
acq_buffer_gray  = []
acq_buffer_bgr   = []
img_count        = 0

binary_vis = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
dist_vis   = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_04.mp4")
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
        acquiring       = True
        acq_buffer_gray = []
        acq_buffer_bgr  = []

    if acquiring:
        gray = cv2.cvtColor(frame_h, cv2.COLOR_BGR2GRAY)
        acq_buffer_gray.append(gray.astype(np.float32))
        acq_buffer_bgr.append(frame_h.astype(np.float32))
        cv2.putText(frame_h, f"Acquiring... {len(acq_buffer_gray)}/{ACQ_FRAMES}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if len(acq_buffer_gray) == ACQ_FRAMES:
            acquiring  = False
            avg_gray   = np.mean(acq_buffer_gray, axis=0).astype(np.uint8)
            avg_bgr    = np.mean(acq_buffer_bgr,  axis=0).astype(np.uint8)
            detected_circles, binary_vis, dist_vis = detect_balls(avg_bgr, avg_gray)

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

    cv2.imshow("Binary (diff+morph)", binary_vis)
    cv2.imshow("Distance transform", dist_vis)
    cv2.imshow("Balls detection", frame_h)

    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(f"{OUTPUT_DIR}/WATERSHED/v08_resultBinary_0{img_count}.png", binary_vis)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v08_resultDist_0{img_count}.png",   dist_vis)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v08_resultOutput_0{img_count}.png", frame_h)
        img_count += 1
        print(f"Saved {img_count}")
        for c in (detected_circles or []):
            print(c)

capture.release()
cv2.destroyAllWindows()

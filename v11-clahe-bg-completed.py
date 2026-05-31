import cv2
import numpy as np
from config import OUTPUT_DIR

H = np.load(f"{OUTPUT_DIR}/homography.npy")
OUTPUT_W, OUTPUT_H = 600, 400

# get scene size
cap_test = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_00.mp4")
clip_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
clip_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_test.release()

# upload the background reference (empty table)
background = cv2.imread(f"{OUTPUT_DIR}/screenshot_00.png")
background_resized = cv2.resize(background, (clip_w, clip_h))
background_warped  = cv2.warpPerspective(background_resized, H, (OUTPUT_W, OUTPUT_H))
background_gray    = cv2.cvtColor(background_warped, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------------------
# Parameters for ball detection

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
MERGE_DIST   = MIN_RADIUS * 2  # circles closer than this across passes = same ball

CLAHE_CLIP = 1.5
CLAHE_TILE = (6, 6)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# mask to avoid the margin outside the playing area
margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN + 2, OUTPUT_H - MARGIN +4), 255, -1)

# ---------------------------------------------------------------------------
# Parameters for cue / ArUco

CUE_MARKER_ID = 4
SHOT_LINE_LEN = 250   # pixels to extend the predicted shot line beyond the white ball
CUE_DEBOUNCE_FRAMES = 15    # if the cue is not visible for n frames, remove the AR overlay

aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params   = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ---------------------------------------------------------------------------
# Functions
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


def find_white_ball(circles, avg_bgr):
    """
    Among all detected circles, return the one that is brightest and least saturated.
    Score = mean(V) - mean(S) in HSV; white ball maximises this.
    """
    if not circles:
        return None

    avg_hsv = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2HSV)
    best_score = -1
    white_ball = None

    for x, y, r in circles:
        mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, r - 2), 255, -1)
        pixels_s = avg_hsv[:, :, 1][mask == 255].astype(float)
        pixels_v = avg_hsv[:, :, 2][mask == 255].astype(float)
        if len(pixels_v) == 0:
            continue
        score = np.mean(pixels_v) - np.mean(pixels_s)
        if score > best_score:
            best_score = score
            white_ball = (x, y, r)

    return white_ball


def detect_cue(frame_original):
    """
    Detect ArUco marker ID 4 in the original (non-warped) frame.
    Returns the marker centre in the scene coordinates (after homography applied), 
    or None.Detecting in the original frame avoids homography distortion of the marker.
    """
    corners, ids, _ = aruco_detector.detectMarkers(frame_original)
    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == CUE_MARKER_ID:
            corner = corners[i][0]  # [4, 2] original coords
            corner_h = cv2.perspectiveTransform(
                corner.reshape(1, -1, 2).astype(np.float32), H)[0] # [4, 2] bird's-eye coords
            cx = int(np.mean(corner_h[:, 0]))
            cy = int(np.mean(corner_h[:, 1]))
            return (cx, cy)

    return None


def clip_line_to_table(x0, y0, dx, dy, length):
    """
    Extend (x0,y0) in direction (dx,dy) for up to `length` pixels,
    stopping at the table play area boundary.
    Returns the endpoint.
    """
    x1 = x0 + dx * length
    y1 = y0 + dy * length
    x1 = max(MARGIN, min(OUTPUT_W - MARGIN, int(x1)))
    y1 = max(MARGIN, min(OUTPUT_H - MARGIN, int(y1)))
    return x1, y1


def draw_overlay(frame_h, detected_circles, circles_from_gray, circles_from_diff, white_ball, cue_center):
    """Draw all AR overlays on the scene frame."""

    # All detected balls (not the white)
    if detected_circles:
        # Green = detected by pass 1 (raw gray)
        for x, y, r in circles_from_gray:
            cv2.circle(frame_h, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame_h, (x, y), 2, (0, 255, 0), -1)
        # Cyan = added by pass 2 (bg subtraction) — likely the green ball or shadows
        for x, y, r in circles_from_diff:
            cv2.circle(frame_h, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame_h, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(frame_h,
                    f"Balls: {len(detected_circles)}  (gray:{len(circles_from_gray)}  diff:+{len(circles_from_diff)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # White ball
    if white_ball:
        wx, wy, wr = white_ball
        cv2.circle(frame_h, (wx, wy), wr, (0, 0, 255), 2)
        cv2.circle(frame_h, (wx, wy), 3, (0, 0, 255), -1)
        cv2.putText(frame_h, "W", (wx - 6, wy - wr - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Cue marker
    # this function works with the assumption that the cue is always pointing
    # to the centre of the white ball, a simplification that makes sense in this simulation
    if cue_center:
        cx, cy = cue_center
        cv2.circle(frame_h, (cx, cy), 3, (0, 220, 255), -1)
        #cv2.putText(frame_h, "CUE", (cx + 8, cy + 5),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    # Shot line: line from white ball in cue direction ---
    if white_ball and cue_center:
        wx, wy, _ = white_ball
        cx, cy = cue_center
        dx = wx - cx
        dy = wy - cy
        dist = np.hypot(dx, dy)
        if dist > 1:
            nx, ny = dx / dist, dy / dist
            # Line from cue to white ball
            cv2.line(frame_h, (cx, cy), (wx, wy), (0, 220, 255), 1)
            # Predicted trajectory: from white ball onward
            ex, ey = clip_line_to_table(wx, wy, nx, ny, SHOT_LINE_LEN)
            cv2.line(frame_h, (wx, wy), (ex, ey),
                            (0, 220, 255), 2)


# ---------------------------------------------------------------------------
# State

detected_circles = None   # None = idle, [] = no balls found, [...] = balls detected
circles_from_gray = []    # for color-coded display
circles_from_diff = []
white_ball       = None
cue_was_seen     = False  # True once ArUco appears after a spacebar detection
acquiring        = False
cue_absent_frames = 0 # for manage the AR overlay view
acq_buffer_gray  = [] # to find the balls
acq_buffer_bgr   = [] # to find the white ball
acq_buffer_diff = [] # to find remaining balls (with binarization)
img_count        = 0

binary_gray = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
binary_diff = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
display_clahe = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_11.mp4") # 28-05 !clip_02, clip_05, clip_11
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

    frame_h = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
    key     = cv2.waitKey(delay) & 0xFF

    # Cue detection: every frame, on original image
    cue_center = detect_cue(frame)

    # --- State machine: cue appearance / disappearance ---
    if cue_center is not None:
        cue_was_seen = True
        cue_absent_frames = 0
    elif cue_was_seen:
        """After cue has been seen the first time, manage the AR overlay view"""
        cue_absent_frames += 1
        if cue_absent_frames >= CUE_DEBOUNCE_FRAMES:
            detected_circles = None
            white_ball = None
            cue_was_seen = False
            cue_absent_frames = 0

    # Spacebar: trigger ball acquisition (also resets cue flag)
    if key == ord(' ') and not acquiring:
        acquiring = True
        acq_buffer_gray = []
        acq_buffer_bgr = []
        acq_buffer_diff = []
        cue_was_seen = False   # fresh start for the new round
        cue_absent_frames = 0

    if acquiring:
        gray = cv2.cvtColor(frame_h, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background_gray)
        acq_buffer_gray.append(gray.astype(np.float32))
        acq_buffer_diff.append(diff.astype(np.float32))
        acq_buffer_bgr.append(frame_h.astype(np.float32))
        cv2.putText(frame_h, f"Acquiring... {len(acq_buffer_gray)}/{ACQ_FRAMES}",
                    (OUTPUT_W - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if len(acq_buffer_gray) == ACQ_FRAMES:
            acquiring = False

            # Pass 1: raw gray average + CLAHE (strong edges on all balls)
            avg_gray = np.mean(acq_buffer_gray, axis=0).astype(np.uint8)
            avg_blur  = cv2.GaussianBlur(avg_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
            avg_clahe = clahe.apply(avg_blur)
            masked_clahe    = cv2.bitwise_and(avg_clahe, avg_clahe, mask=margin_mask)
            circles_from_gray = run_hough(masked_clahe, PARAM2_GRAY, MIN_DIST_1)

            # Pass 2: background subtraction (sensitive to low-contrast balls)
            avg_diff = np.mean(acq_buffer_diff, axis=0).astype(np.uint8)
            avg_diff = cv2.GaussianBlur(avg_diff, (BLUR_KERNEL, BLUR_KERNEL), 0)
            # close fills gaps within blobs; open removes thin shadow artifacts
            avg_diff = cv2.morphologyEx(avg_diff, cv2.MORPH_CLOSE, kernel_close)
            avg_diff = cv2.morphologyEx(avg_diff, cv2.MORPH_OPEN,  kernel_open)
            binary_diff = cv2.bitwise_and(avg_diff, avg_diff, mask=margin_mask)
            circles_from_diff_raw = run_hough(binary_diff, PARAM2_DIFF, MIN_DIST_2)

            avg_bgr  = np.mean(acq_buffer_bgr,  axis=0).astype(np.uint8)

            # Merge: keep pass-1 results, add pass-2 detections not yet covered
            detected_circles, circles_from_diff = merge_circles(
                circles_from_gray, circles_from_diff_raw, MERGE_DIST
            )
            """
            if raw_circles is not None:
                detected_circles = [(int(x), int(y), int(r))
                                    for x, y, r in np.round(raw_circles[0])]
            else:
                detected_circles = []
            """

            # White ball identification
            white_ball = find_white_ball(detected_circles, avg_bgr)
            display_clahe = masked_clahe

    # --- Draw overlays ---
    draw_overlay(frame_h, detected_circles, circles_from_gray, circles_from_diff, white_ball, cue_center)

    if detected_circles is None and not acquiring:
        cv2.putText(frame_h, "Press SPACEBAR to detect balls",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
    elif detected_circles == [] and not acquiring:
        cv2.putText(frame_h, "No balls found!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("CLAHE", display_clahe)
    cv2.imshow("Snooker Helper", frame_h)

    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v11_resultCLAHE_0{img_count}.png",  display_clahe)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v11_resultOutput_0{img_count}.png", frame_h)
        img_count += 1
        print(f"Saved {img_count}")
        if white_ball:
            print(f"White ball: {white_ball}")
        if cue_center:
            print(f"Cue center: {cue_center}")
        for c in (detected_circles or []):
            print(c)

capture.release()
cv2.destroyAllWindows()
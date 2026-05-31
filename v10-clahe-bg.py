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
# Parameters — ball detection (same as v09)
# ---------------------------------------------------------------------------
BLUR_KERNEL = 3
DP          = 1.3
PARAM1      = 50
PARAM2      = 17
MIN_RADIUS  = 14
MAX_RADIUS  = 17
MIN_DIST    = MIN_RADIUS * 1.5
MARGIN      = 45
ACQ_FRAMES  = 15

CLAHE_CLIP = 1.5
CLAHE_TILE = (6, 6)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN, OUTPUT_H - MARGIN), 255, -1)

# ---------------------------------------------------------------------------
# Parameters — cue / ArUco
# ---------------------------------------------------------------------------
CUE_MARKER_ID = 4
SHOT_LINE_LEN = 250   # pixels to extend the predicted shot line beyond the white ball

aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params   = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_white_ball(circles, avg_bgr):
    """
    Among all detected circles, return the one that is brightest and least saturated.
    Score = mean(V) - mean(S) in HSV; white ball maximises this.
    """
    if not circles:
        return None

    avg_hsv    = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2HSV)
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
    Returns the marker centre in bird's-eye coordinates, or None.
    Detecting in the original frame avoids homography distortion of the marker.
    """
    corners, ids, _ = aruco_detector.detectMarkers(frame_original)
    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == CUE_MARKER_ID:
            corner = corners[i][0]                                  # [4, 2] original coords
            corner_h = cv2.perspectiveTransform(
                corner.reshape(1, -1, 2).astype(np.float32), H
            )[0]                                                     # [4, 2] bird's-eye coords
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


def draw_overlay(frame_h, detected_circles, white_ball, cue_center):
    """Draw all AR overlays on the bird's-eye frame."""

    # --- All detected balls: green ---
    if detected_circles:
        for x, y, r in detected_circles:
            cv2.circle(frame_h, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame_h, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(frame_h, f"Balls: {len(detected_circles)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- White ball: red circle + label ---
    if white_ball:
        wx, wy, wr = white_ball
        cv2.circle(frame_h, (wx, wy), wr, (0, 0, 255), 2)
        cv2.circle(frame_h, (wx, wy), 3, (0, 0, 255), -1)
        cv2.putText(frame_h, "W", (wx - 6, wy - wr - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- Cue marker: yellow dot ---
    # this function works with the assumption that the cue is always pointing
    # to the centre of the white ball, a simplification that makes sense in this simulation
    if cue_center:
        cx, cy = cue_center
        cv2.circle(frame_h, (cx, cy), 3, (0, 220, 255), -1)
        cv2.putText(frame_h, "CUE", (cx + 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    # --- Shot line: cyan arrow from white ball in cue direction ---
    if white_ball and cue_center:
        wx, wy, _ = white_ball
        cx, cy    = cue_center
        dx = wx - cx
        dy = wy - cy
        dist = np.hypot(dx, dy)
        if dist > 1:
            nx, ny = dx / dist, dy / dist
            # Line from cue to white ball (dashed-feel: thin)
            cv2.line(frame_h, (cx, cy), (wx, wy), (0, 220, 255), 1)
            # Predicted trajectory: from white ball onward
            ex, ey = clip_line_to_table(wx, wy, nx, ny, SHOT_LINE_LEN)
            cv2.line(frame_h, (wx, wy), (ex, ey),
                            (0, 220, 255), 2)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
detected_circles = None   # None = idle, [] = no balls found, [...] = balls detected
white_ball       = None
cue_was_seen     = False  # True once ArUco appears after a spacebar detection
acquiring        = False
acq_buffer_gray  = []
acq_buffer_bgr   = []
img_count        = 0

display_clahe = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_03.mp4")
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

    # --- Cue detection: every frame, on original image ---
    cue_center = detect_cue(frame)

    # --- State machine: cue appearance / disappearance ---
    if cue_center is not None:
        # Marker visible: arm the "was seen" flag
        cue_was_seen = True
    elif cue_was_seen: 
        # Marker WAS visible but is now gone → shot taken → reset everything
        detected_circles = None
        white_ball       = None
        cue_was_seen     = False

    # --- Spacebar: trigger ball acquisition (also resets cue flag) ---
    if key == ord(' ') and not acquiring:
        acquiring       = True
        acq_buffer_gray = []
        acq_buffer_bgr  = []
        cue_was_seen    = False   # fresh start for the new round

    if acquiring:
        gray = cv2.cvtColor(frame_h, cv2.COLOR_BGR2GRAY)
        acq_buffer_gray.append(gray.astype(np.float32))
        acq_buffer_bgr.append(frame_h.astype(np.float32))
        cv2.putText(frame_h, f"Acquiring... {len(acq_buffer_gray)}/{ACQ_FRAMES}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if len(acq_buffer_gray) == ACQ_FRAMES:
            acquiring = False

            avg_gray = np.mean(acq_buffer_gray, axis=0).astype(np.uint8)
            avg_bgr  = np.mean(acq_buffer_bgr,  axis=0).astype(np.uint8)

            # Ball detection (CLAHE + HoughCircles)
            avg_blur  = cv2.GaussianBlur(avg_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
            avg_clahe = clahe.apply(avg_blur)
            masked    = cv2.bitwise_and(avg_clahe, avg_clahe, mask=margin_mask)

            raw_circles = cv2.HoughCircles(
                masked, cv2.HOUGH_GRADIENT,
                dp=DP, minDist=MIN_DIST,
                param1=PARAM1, param2=PARAM2,
                minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
            )

            if raw_circles is not None:
                detected_circles = [(int(x), int(y), int(r))
                                    for x, y, r in np.round(raw_circles[0])]
            else:
                detected_circles = []

            # White ball identification
            white_ball = find_white_ball(detected_circles, avg_bgr)
            display_clahe = masked

    # --- Draw overlays ---
    draw_overlay(frame_h, detected_circles, white_ball, cue_center)

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
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v10_resultCLAHE_0{img_count}.png",  display_clahe)
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v10_resultOutput_0{img_count}.png", frame_h)
        img_count += 1
        print(f"--- Saved {img_count} ---")
        if white_ball:
            print(f"White ball: {white_ball}")
        if cue_center:
            print(f"Cue center (bird's-eye): {cue_center}")
        for c in (detected_circles or []):
            print(c)

capture.release()
cv2.destroyAllWindows()

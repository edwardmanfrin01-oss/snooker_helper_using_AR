import cv2
import numpy as np
from config import OUTPUT_DIR

H = np.load(f"{OUTPUT_DIR}/homography.npy")
OUTPUT_W, OUTPUT_H = 600, 400

CLIP = f"{OUTPUT_DIR}/clip_06.mp4"

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
# Parameters — ball detection
# ---------------------------------------------------------------------------
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

CLAHE_CLIP = 1.5
CLAHE_TILE = (8, 8)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

margin_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
cv2.rectangle(margin_mask, (MARGIN, MARGIN),
              (OUTPUT_W - MARGIN + 2, OUTPUT_H - MARGIN + 5), 255, -1)

# ---------------------------------------------------------------------------
# Parameters — cue / ArUco
# ---------------------------------------------------------------------------
CUE_MARKER_ID       = 4
SHOT_LINE_LEN       = 350   # max pixels for trajectory lines
CUE_DEBOUNCE_FRAMES = 15

aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params   = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

#
# Parameters for colors
#
WHITE = (255,255,255)
BLACK = (0,0,0)

def writing(text, location):
    cv2.putText(frame_h, text,
                location, cv2.FONT_HERSHEY_DUPLEX, 0.7, BLACK, 5)
    cv2.putText(frame_h, text,
                location, cv2.FONT_HERSHEY_DUPLEX, 0.7, WHITE, 2)
# ---------------------------------------------------------------------------
# Ball detection helpers
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
    merged = list(primary)
    added  = []
    for cx, cy, cr in secondary:
        if all(np.hypot(cx - mx, cy - my) >= dist_threshold for mx, my, _ in merged):
            merged.append((cx, cy, cr))
            added.append((cx, cy, cr))
    return merged, added


def find_white_ball(circles, avg_bgr):
    """Brightest + least saturated circle in HSV = white ball."""
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

# ---------------------------------------------------------------------------
# Cue detection
# ---------------------------------------------------------------------------

def detect_cue(frame_original):
    corners, ids, _ = aruco_detector.detectMarkers(frame_original)
    if ids is None:
        return None
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == CUE_MARKER_ID:
            corner   = corners[i][0]
            corner_h = cv2.perspectiveTransform(
                corner.reshape(1, -1, 2).astype(np.float32), H)[0]
            cx = int(np.mean(corner_h[:, 0]))
            cy = int(np.mean(corner_h[:, 1]))
            return (cx, cy)
    return None

# ---------------------------------------------------------------------------
# Trajectory physics
# ---------------------------------------------------------------------------

def ray_vs_ball(px, py, nx, ny, bx, by, br, wr, eps=1.0):
    """
    Intersection of a ray (origin px,py; unit direction nx,ny) with a ball
    at (bx,by) of radius br. The white ball has radius wr, so collision
    happens when centre-to-centre distance = wr + br.
    Returns the smallest positive t (> eps) or None.
    eps is made to avoid collision of the white with itself
    """
    R  = wr + br
    dx = px - bx
    dy = py - by
    b  = dx * nx + dy * ny
    c  = dx * dx + dy * dy - R * R
    disc = b * b - c
    if disc < 0:
        return None
    sq   = np.sqrt(disc)
    t1, t2 = -b - sq, -b + sq
    if t1 > eps:
        return t1
    if t2 > eps:
        return t2
    return None


def ray_vs_walls(px, py, nx, ny, eps=1e-6):
    """
    Intersection of a ray with the four table walls (at MARGIN boundaries).
    Returns (t, axis) where axis is 'x' (left/right) or 'y' (top/bottom),
    or (inf, None) if no valid hit.
    """
    x_min, x_max = MARGIN, OUTPUT_W - MARGIN
    y_min, y_max = MARGIN, OUTPUT_H - MARGIN
    t_min, axis  = float('inf'), None

    candidates = [
        (nx,  x_max - px, 'x'),   # right wall
        (-nx, px - x_min, 'x'),   # left wall
        (ny,  y_max - py, 'y'),   # bottom wall
        (-ny, py - y_min, 'y'),   # top wall
    ]
    for speed, gap, ax in candidates:
        if speed > eps:
            t = gap / speed
            if t > eps and t < t_min:
                # check the perpendicular coordinate is within bounds
                hx = px + (gap / speed if ax == 'x' else t) * nx if ax == 'y' else None
                hy = py + (gap / speed if ax == 'y' else t) * ny if ax == 'x' else None
                if ax == 'x':
                    hy = py + t * ny
                    if y_min <= hy <= y_max:
                        t_min, axis = t, ax
                else:
                    hx = px + t * nx
                    if x_min <= hx <= x_max:
                        t_min, axis = t, ax

    return t_min, axis


def draw_trajectory(frame_h, white_ball, cue_center, detected_circles):
    """
    Physics-based trajectory overlay:

    Case 1 — ball hit first:
        • Cyan line : white ball current pos → white ball impact pos
        • Cyan ghost: circle at white ball impact pos (shows where it stops)
        • Orange line: target ball current pos → target ball exit direction
                       (direction = line through the two centres at impact)

    Case 2 — wall hit first (no ball in the way):
        • Cyan line : white ball → wall contact point
        • Cyan dot  : bounce point marker
        • Dimmer line: reflected trajectory (one bounce only, no further checks)
    """
    if not white_ball or not cue_center:
        return

    wx, wy, wr = white_ball
    cx, cy     = cue_center

    dx, dy = wx - cx, wy - cy
    dist   = np.hypot(dx, dy)
    if dist < 1:
        return
    nx, ny = dx / dist, dy / dist

    # All balls except the white one
    others = [b for b in detected_circles if b != white_ball]

    # --- find closest ball in the shot direction ---
    t_ball  = float('inf')
    hit_ball = None
    for bx, by, br in others:
        t = ray_vs_ball(wx, wy, nx, ny, bx, by, br, wr)
        if t is not None and t < t_ball:
            t_ball   = t
            hit_ball = (bx, by, br)

    # --- find wall hit ---
    t_wall, wall_axis = ray_vs_walls(wx, wy, nx, ny)

    # ---------------------------------------------------------------
    # Case 1: ball hit before wall
    # ---------------------------------------------------------------
    if hit_ball is not None and t_ball <= t_wall:
        bx, by, br = hit_ball
        imp_x = int(wx + nx * t_ball)
        imp_y = int(wy + ny * t_ball)

        # White ball: from current pos to impact pos
        cv2.line(frame_h, (wx, wy), (imp_x, imp_y), (0, 220, 255), 2)
        # Ghost circle: shows where the white ball stops
        cv2.circle(frame_h, (imp_x, imp_y), wr, (0, 220, 255), 1)

        # Target ball direction: normalised vector from impact-white-centre to target-centre
        tdx = bx - imp_x
        tdy = by - imp_y
        td  = np.hypot(tdx, tdy)
        if td > 1:
            tnx, tny = tdx / td, tdy / td
            # Extend to wall (or fallback to fixed length)
            t_tw, _ = ray_vs_walls(bx, by, tnx, tny)
            if t_tw < float('inf'):
                tex = int(bx + tnx * t_tw)
                tey = int(by + tny * t_tw)
            else:
                tex = int(bx + tnx * SHOT_LINE_LEN)
                tey = int(by + tny * SHOT_LINE_LEN)
            # Highlight targeted ball + draw its trajectory
            cv2.circle(frame_h, (int(bx), int(by)), br + 2, (0, 140, 255), 2)
            cv2.line(frame_h, (int(bx), int(by)), (tex, tey), (0, 140, 255), 2)

    # ---------------------------------------------------------------
    # Case 2: wall hit before any ball
    # ---------------------------------------------------------------
    elif t_wall < float('inf'):
        wx1 = int(wx + nx * t_wall)
        wy1 = int(wy + ny * t_wall)

        # White ball trajectory to wall
        cv2.line(frame_h, (wx, wy), (wx1, wy1), (0, 220, 255), 2)
        cv2.circle(frame_h, (wx1, wy1), 4, (0, 220, 255), -1)   # bounce dot

        # Reflected direction (one bounce)
        rnx = -nx if wall_axis == 'x' else nx
        rny = -ny if wall_axis == 'y' else ny

        # Extend reflected ray to next wall
        t_wall2, _ = ray_vs_walls(wx1, wy1, rnx, rny)
        if t_wall2 < float('inf'):
            wx2 = int(wx1 + rnx * t_wall2)
            wy2 = int(wy1 + rny * t_wall2)
        else:
            wx2 = int(wx1 + rnx * SHOT_LINE_LEN)
            wy2 = int(wy1 + rny * SHOT_LINE_LEN)

        cv2.line(frame_h, (wx1, wy1), (wx2, wy2), (0, 170, 200), 1)  # dimmer after bounce

# ---------------------------------------------------------------------------
# AR overlay (balls + white ball + cue + trajectory)
# ---------------------------------------------------------------------------

def draw_overlay(frame_h, detected_circles, circles_from_gray, circles_from_diff,
                 white_ball, cue_center):

    if detected_circles:
        for x, y, r in circles_from_gray:
            cv2.circle(frame_h, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame_h, (x, y), 2, (0, 255, 0), -1)
        for x, y, r in circles_from_diff:
            cv2.circle(frame_h, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame_h, (x, y), 3, (255, 0, 0), -1)
        writing(f"Balls: {len(detected_circles)} (gray:{len(circles_from_gray)}  diff:{len(circles_from_diff)})",(10, 25))

    if white_ball:
        wx, wy, wr = white_ball
        cv2.circle(frame_h, (wx, wy), wr, (0, 0, 255), 2)
        cv2.circle(frame_h, (wx, wy), 3,  (0, 0, 255), -1)
        cv2.putText(frame_h, "W", (wx - 6, wy - wr - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if cue_center:
        cv2.circle(frame_h, cue_center, 3, (0, 220, 255), -1)

    # Thin line from cue marker to white ball (aiming guide)
    if white_ball and cue_center:
        cv2.line(frame_h, cue_center, (white_ball[0], white_ball[1]), (0, 220, 255), 1)

    # Physics-based trajectory
    draw_trajectory(frame_h, white_ball, cue_center, detected_circles or [])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
detected_circles  = None
circles_from_gray = []
circles_from_diff = []
white_ball        = None
cue_was_seen      = False
cue_absent_frames = 0
acquiring         = False
acq_buffer_gray   = []
acq_buffer_bgr    = []
acq_buffer_diff   = []
img_count         = 0

binary_gray   = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
binary_diff   = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
display_clahe = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)

capture = cv2.VideoCapture(CLIP)
#capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

fps   = 60
delay = int(1000 / fps)

# 
# save clip
# Output parameters
#
clip_count = 1
rec = False
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(f"{OUTPUT_DIR}/RESULTS/v12_rec_0{clip_count}.mp4", fourcc, delay, (OUTPUT_W, OUTPUT_H))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_h = cv2.warpPerspective(frame, H, (OUTPUT_W, OUTPUT_H))
    key = cv2.waitKey(delay) & 0xFF

    # Cue detection every frame
    cue_center = detect_cue(frame)

    # State machine with debounce
    if cue_center is not None:
        cue_was_seen      = True
        cue_absent_frames = 0
    elif cue_was_seen:
        cue_absent_frames += 1
        if cue_absent_frames >= CUE_DEBOUNCE_FRAMES:
            detected_circles  = None
            white_ball        = None
            cue_was_seen      = False
            cue_absent_frames = 0

    # Spacebar triggers acquisition
    if key == ord(' ') and not acquiring:
        acquiring         = True
        acq_buffer_gray   = []
        acq_buffer_bgr    = []
        acq_buffer_diff   = []
        cue_was_seen      = False
        cue_absent_frames = 0

    if acquiring:
        gray = cv2.cvtColor(frame_h, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background_gray)
        acq_buffer_gray.append(gray.astype(np.float32))
        acq_buffer_diff.append(diff.astype(np.float32))
        acq_buffer_bgr.append(frame_h.astype(np.float32))
        writing(f"Acquiring... {len(acq_buffer_gray)}/{ACQ_FRAMES}",(int(OUTPUT_W/2) - 120, int(OUTPUT_H/2)))

        if len(acq_buffer_gray) == ACQ_FRAMES:
            acquiring = False

            avg_gray  = np.mean(acq_buffer_gray, axis=0).astype(np.uint8)
            avg_bgr   = np.mean(acq_buffer_bgr,  axis=0).astype(np.uint8)

            # Pass 1: CLAHE on gray
            avg_blur  = cv2.GaussianBlur(avg_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
            avg_clahe = clahe.apply(avg_blur)
            masked_clahe      = cv2.bitwise_and(avg_clahe, avg_clahe, mask=margin_mask)
            circles_from_gray = run_hough(masked_clahe, PARAM2_GRAY, MIN_DIST_1)

            # Pass 2: background subtraction
            avg_diff = np.mean(acq_buffer_diff, axis=0).astype(np.uint8)
            avg_diff = cv2.GaussianBlur(avg_diff, (BLUR_KERNEL, BLUR_KERNEL), 0)
            avg_diff = cv2.morphologyEx(avg_diff, cv2.MORPH_CLOSE, kernel_close)
            avg_diff = cv2.morphologyEx(avg_diff, cv2.MORPH_OPEN,  kernel_open)
            binary_diff = cv2.bitwise_and(avg_diff, avg_diff, mask=margin_mask)
            circles_from_diff_raw = run_hough(binary_diff, PARAM2_DIFF, MIN_DIST_2)

            detected_circles, circles_from_diff = merge_circles(
                circles_from_gray, circles_from_diff_raw, MERGE_DIST)

            white_ball = find_white_ball(detected_circles, avg_bgr)
            display_clahe = masked_clahe

    # Draw
    draw_overlay(frame_h, detected_circles, circles_from_gray, circles_from_diff,
                 white_ball, cue_center)

    if detected_circles is None and not acquiring:
        writing("Press SPACEBAR to detect balls",(int(OUTPUT_W/2) - 180, int(OUTPUT_H/2)))
    elif detected_circles == [] and not acquiring:
        writing("No balls found!", (int(OUTPUT_W/2) - 100, int(OUTPUT_H/2)))

    cv2.imshow("CLAHE", display_clahe)
    cv2.imshow("Snooker Helper", frame_h)
    cv2.imshow("avg_diff", binary_diff)

    if rec:
        writer.write(frame_h)

    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(f"{OUTPUT_DIR}/RESULTS/v12_resultOutput_0{img_count}.png", frame_h)
        img_count += 1
        print(f"Saved {img_count}")
        if white_ball:
            print(f"White ball: {white_ball}")
        if cue_center:
            print(f"Cue: {cue_center}")
    if key == ord('r'):
        # save clip
        if not rec:
            rec = True
        else:
            rec = False
            writer.release()

capture.release()
writer.release()
cv2.destroyAllWindows()

import cv2
import os
from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#capture.set(cv2.CAP_PROP_FPS, 30)

w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {w}x{h}")

fps = capture.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
screenshot_count = 8
clip_count = 11
recording = False
writer = None

while True:
    ret, frame = capture.read()
    if not ret:
        break

    display = frame.copy()
    if recording:
        writer.write(frame)
        cv2.circle(display, (20, 20), 10, (0, 0, 255), -1)
        cv2.putText(display, "REC", (35, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(display, "Press R for recording, S for screenshot", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Recording", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        if not recording:
            clip_path = os.path.join(OUTPUT_DIR, f"clip_{clip_count:02d}.mp4")
            writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
            recording = True
            print(f"Recording: {clip_path} @ {fps}fps")
        else:
            writer.release()
            writer = None
            recording = False
            print(f"Clip {clip_count:02d} saved.")
            clip_count += 1
    elif key == ord('s'):
        screenshot_path = os.path.join(OUTPUT_DIR, f"screenshot_{screenshot_count:02d}.png")
        cv2.imwrite(screenshot_path, frame)
        print(f"screenshot_{screenshot_count:02d}.png saved")
        screenshot_count += 1
    elif key == ord('q'):
        break

if writer:
    writer.release()
capture.release()
cv2.destroyAllWindows()
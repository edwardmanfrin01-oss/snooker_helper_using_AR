# Find the homography, to map the Aruco Markers' centers (src_pts) to 
# to the edges in the new scene (dst_pts)
# Must be runned everytime the scene is moved. 

import cv2
import numpy as np
from config import OUTPUT_DIR


OUTPUT_W, OUTPUT_H = 600, 400

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#capture = cv2.VideoCapture(f"{OUTPUT_DIR}/clip_00.mp4")

while True:
    ret, frame = capture.read()
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        marker_centers = {}
        for i, mid in enumerate(ids.flatten()):
            if mid in [0, 1, 2, 3]:
                marker_centers[mid] = corners[i][0].mean(axis=0)

        if len(marker_centers) == 4:
            cv2.putText(frame, "All 4 markers found - press 's' to save",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                src_pts = np.float32([
                                        marker_centers[3],  # top-left
                                        marker_centers[2],  # top-right
                                        marker_centers[1],  # bottom-right
                                        marker_centers[0],  # bottom-left
                                    ])
                dst_pts = np.float32([
                                        [0,0],
                                        [OUTPUT_W,0],
                                        [OUTPUT_W,OUTPUT_H],
                                        [0,OUTPUT_H]
                                    ])
                H, _ = cv2.findHomography(src_pts, dst_pts)
                np.save(f"{OUTPUT_DIR}/homography.npy", H)
                print(f"Homography saved in {OUTPUT_DIR}/homography.npy")
                break
        else:
            cv2.putText(frame, f"Markers in the capture: {list(marker_centers.keys())}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Scene calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
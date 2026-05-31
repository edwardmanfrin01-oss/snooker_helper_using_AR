# generate the aruco markers to put on the corners of the table

import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

for marker_id in range(5):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 500)
    cv2.imwrite(f"PROJECT/images/marker_{marker_id}_500.png", marker_img)
    print(f"Marker {marker_id} salvato.")



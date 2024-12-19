import cv2
from face_encoding import Face
import time

custom_encoder = Face()
custom_encoder.load_encoding_images("Dataset/")
unique_names = set()
attendance_log = open("attendance_log.txt", "a")

re_enroll_interval = 30 
last_re_enroll_time = time.time()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face_areas, recognized_names = custom_encoder.detect_known_faces(frame)
    for face_coords, found_name in zip(face_areas, recognized_names):
        t, r, b, l = face_coords[0], face_coords[1], face_coords[2], face_coords[3]
        if found_name not in unique_names:
            unique_names.add(found_name)
            attendance_log.write(found_name + "\n")
        if time.time() - last_re_enroll_time > re_enroll_interval:
            custom_encoder.load_encoding_images("Dataset/") 
            last_re_enroll_time = time.time()
        cv2.putText(frame, found_name, (l, t - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.rectangle(frame, (l, t), (r, b), (0,   0, 200), 4)

    cv2.imshow("DataFlair", frame)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
attendance_log.close()


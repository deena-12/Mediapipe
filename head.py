import cv2
import mediapipe as mp

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)
class_names=['person,animals,bike,car,bycycle,boat,aeroplane,']
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Skipping empty frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            left = face_landmarks.landmark[234]
            right = face_landmarks.landmark[454]

            # Pixel coordinates
            fx, fy = int(forehead.x * w), int(forehead.y * h)
            cx, cy = int(chin.x * w), int(chin.y * h)
            lx, ly = int(left.x * w), int(left.y * h)
            rx, ry = int(right.x * w), int(right.y * h)

            # Calculate estimated top of head box
            face_height = cy - fy
            box_height = int(face_height * 0.6)  
            box_top_y = fy - box_height  
            box_bottom_y = fy 

            # Estimate horizontal width
            box_width = int((rx - lx) * 0.6)
            box_left_x = fx - box_width // 2
            box_right_x = fx + box_width // 2

            # Draw red rectangle around top of head
            cv2.rectangle(frame, (box_left_x, box_top_y), (box_right_x, box_bottom_y), (0, 0, 255), 2)
            cv2.putText(frame, "Top Head", (box_left_x, box_top_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Top of Head Only", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



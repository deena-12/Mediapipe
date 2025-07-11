
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('Mediapipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



"""
import cv2
import mediapipe as mp


mp_face=mp.solutions.face_detection
mp_hands=mp.solutions.hands     
mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

face_detection=mp_face.FaceDetection(min_detection_confidence=0.5)
hands=mp_hands.Hands(min_detection_confidence=0.5,max_num_hands=2)
pose=mp_pose.Pose(min_detection_confidence=0.5)

cv2.namedWindow("Multi-Track Mediapipe", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Multi-Track Mediapipe", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while cap.isOpened():
    success,frame=cap.read()
    if not success:                            
        print("skiping empty frame")
        continue

    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 

    face_results=face_detection.process(rgb)
    hand_results=hands.process(rgb)
    pose_results=pose.process(rgb)

    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame,detection)

    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,handLms,mp_hands.HAND_CONNECTIONS)          
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Multi-Track Mediapipe",frame)
    if cv2.waitKey(5) & 0xFF==ord('q'):
        break
             
cap.release()
cv2.destroyAllWindows()
    """        


 #full face covered sample 
"""
import cv2
import mediapipe as mp

# Initialize Mediapipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize models
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2)
pose = mp_pose.Pose(min_detection_confidence=0.5)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process all models
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    h, w, _ = frame.shape

    # Face Mesh: Show eyes with label
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw all facial landmarks
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Right Eye (landmark 33)
            eye = face_landmarks.landmark[33]
            x, y = int(eye.x * w), int(eye.y * h)
            cv2.putText(frame, 'Eye', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # Hand Tracking
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Label the wrist point (landmark 0)
            wrist = handLms.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, 'Hand', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Pose Detection
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show output
    cv2.imshow("Mediapipe with Labels", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""
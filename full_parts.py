
"""
import cv2
import mediapipe as mp
mp_pose=mp.solutions.pose
mp_drawings=mp.solutions.drawing_utils
pose=mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.5)
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
while cap.isOpened():
    success,frame=cap.read()
    if not success:
        print("skipping empty frame.")
        continue
    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=pose.process(rgb)
    if results.pose_landmarks:
        mp_drawings.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        h , w,_=frame.shape
        landmarks_to_label={
             "Nose": 0,
            "Left Eye": 2,
            "Right Eye": 5,
            "Left Ear": 7,
            "Right Ear": 8,
            "Left Shoulder": 11,
            "Right Shoulder": 12,
            "Left Elbow": 13,
            "Right Elbow": 14,
            "Left Wrist": 15,
            "Right Wrist": 16,
            "Left Hip": 23,
            "Right Hip": 24,
            "Left Knee": 25,
            "Right Knee": 26, 
            "Left Ankle": 27,
            "Right Ankle": 28  
        }
        for name, idx in landmarks_to_label.items():
            lm = results.pose_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Full Body Pose Detection", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""


        
#full parts

import cv2
import mediapipe as mp

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)   
                                                      
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Skipping empty frame.")
        continue
                                                                
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)
    if results.pose_landmarks:
        # Draw full body skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get frame dimensions
        h, w, _ = frame.shape

        # Define specific landmark indices and names
        landmarks_to_label = {
            "Nose": 0,
            "Left Eye": 2,
            "Right Eye": 5,
            "Left Ear": 7,
            "Right Ear": 8,
            "Left Shoulder": 11,
            "Right Shoulder": 12,
            "Left Elbow": 13,
            "Right Elbow": 14,
            "Left Wrist": 15,
            "Right Wrist": 16,
            "Left Hip": 23,
            "Right Hip": 24,
            "Left Knee": 25,
            "Right Knee": 26,
            "Left Ankle": 27,
            "Right Ankle": 28
        }

        for name, idx in landmarks_to_label.items():
            lm = results.pose_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Full Body Pose Detection", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



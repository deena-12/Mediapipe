import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")  

# Load your sample video       
cap = cv2.VideoCapture("videos/people.mp4") 

# Create a full screen window
cv2.namedWindow("Top Head Detection & Count", cv2.WINDOW_NORMAL)             
cv2.resizeWindow("Top Head Detection & Count", 1280, 720)
                                                               
while cap.isOpened():         
    ret, frame = cap.read()                                 
    if not ret:              
        break                                                              
    # Run YOLOv8 inference
    results = model(frame)

    # Filter only "person"           
    person_boxes = [                 
        box for box in results[0].boxes 
        if int(box.cls[0]) == 0 and box.conf[0] > 0.4
    ]                          
    # Draw top head box 
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])                              
        # Define the top 30% of bounding box as the head region
        head_top = y1                                                                                                     
        head_bottom = y1 + int((y2 - y1) * 0.3)                    
        head_left = x1 + int((x2 - x1) * 0.2)                                                                    
        head_right = x2 - int((x2 - x1) * 0.2)  

        # Draw red box for top of the head
        cv2.rectangle(frame, (head_left, head_top), (head_right, head_bottom), (0, 0, 255), 2)
        cv2.putText(frame, "Person", (head_left, head_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # Show head count on top-left
    cv2.putText(frame, f"Head Count: {len(person_boxes)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
    # Resize to fit screen
    frame_resized = cv2.resize(frame, (1920, 1080))  # You can change to (1920, 1080) if needed
    cv2.imshow("Top Head Detection & Count", frame_resized)

    # Press 'q' to exit                      
    if cv2.waitKey(1) & 0xFF == ord('q'):                                                                       
        break                            
                
# Release resources               
cap.release()
cv2.destroyAllWindows()



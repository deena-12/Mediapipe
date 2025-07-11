import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet
from ultralytics import YOLO

# --- Load your own sample images ---
SAMPLE_DIR = "test_images"  # Folder containing your own test images

# --- Initialize models ---
print("🔍 Loading YOLOv8...")                 
yolo = YOLO("yolov8n.pt")

print("🔍 Loading FaceNet...")
embedder = FaceNet()                   

# --- Helper functions ---
def detect_face(img):
    results = yolo(img)
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        return img[y1:y2, x1:x2]
    return None

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32") / 255.0
    return face_img

def get_embedding(face_img):
    return embedder.embeddings([face_img])[0]

# --- Build embedding database from sample images ---
print(" Building embedding database from test_images/")
embedding_db = {}
test_images = []

for file in os.listdir(SAMPLE_DIR):
    path = os.path.join(SAMPLE_DIR, file)
    name, _ = os.path.splitext(file) 
    img = cv2.imread(path)
    if img is None:
        continue
    face = detect_face(img)
    if face is None:
        print(f"No face found in {file}, skipping.")
        continue
    face = preprocess_face(face)
    embedding = get_embedding(face)
    embedding_db[name] = [embedding]
    test_images.append((name, img))

print(f"✅ Database built with {len(embedding_db)} people.")

# --- Recognition ---
def recognize_face(img_bgr, db, threshold=0.6):
    face = detect_face(img_bgr)
    if face is None:
        return "No face", 0
    face = preprocess_face(face)
    emb = get_embedding(face)

    best_match = "Unknown"
    best_score = -1

    for name, emb_list in db.items():
        for known_emb in emb_list:
            score = cosine_similarity([emb], [known_emb])[0][0]
            if score > best_score:
                best_score = score
                best_match = name if score > threshold else "Unknown"
    return best_match, best_score

# --- Test all faces ---
print("🧪 Running recognition tests...")
correct = 0
total = 0

for real_name, img in tqdm(test_images):
    predicted_name, score = recognize_face(img, embedding_db)
    total += 1
    if predicted_name == real_name:
        correct += 1
    print(f"[{total}] Real: {real_name} | Predicted: {predicted_name} | Score: {score:.2f}")

print(f"\n🎯 Accuracy: {correct}/{total} = {(correct / total) * 100:.2f}%")



import os
import cv2
import tarfile
import urllib.request
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Download and extract LFW dataset if not present
URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
TGZ = "lfw-deepfunneled.tgz"
DATASET_DIR = "lfw_funneled"
    

if not os.path.isdir(DATASET_DIR):
    print("⬇️ Downloading LFW dataset...")
    urllib.request.urlretrieve(URL, TGZ)
    print("📂 Extracting…")  
    with tarfile.open(TGZ) as tar:
        tar.extractall()                       
    print("✅ Dataset ready at", DATASET_DIR)
# Step 2: Initialize YOLO and FaceNet models
yolo = YOLO("yolov8n.pt")
embedder = FaceNet()                
  
# Step 3: Build face embedding database           
face_db = {}
print("📸 Building embedding database...")   

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir)[:1]:  # Take only 1 image per person for demo
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        embeddings = embedder.embeddings([img])
        face_db[person_name] = embeddings[0]
        
# Step 4: Real-time camera recognition 
print("🎥 Starting webcam for face recognition...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect faces
    results = yolo(frame, classes=[0], conf=0.5)[0]  # class 0 = person
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        try:
            embedding = embedder.embeddings([face])[0]
        except:
            continue
        finally:
            print('thereisa known to debugg the error ')
        # Compare with database
        name = "Unknown"
        max_sim = 0.5 
        for db_name, db_embedding in face_db.items():
            sim = cosine_similarity([embedding], [db_embedding])[0][0]
            if sim > max_sim:
                max_sim = sim
                name = db_name
    
        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
        
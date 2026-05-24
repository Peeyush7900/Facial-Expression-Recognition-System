import os
import cv2
import numpy as np
import random
from pathlib import Path

# Target directories
DATA_DIR = Path(__file__).parent.parent / "data"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def create_synthetic_face(emotion: str) -> np.ndarray:
    """
    Generates a 48x48 grayscale synthetic face image with features representing the specified emotion.
    """
    # Create black background
    img = np.zeros((48, 48), dtype=np.uint8)
    
    # Base face shape (light gray circle in the center)
    # Add a bit of random variation in color/size to simulate realistic datasets
    face_color = random.randint(180, 220)
    face_radius = random.randint(19, 21)
    cv2.circle(img, (24, 24), face_radius, face_color, -1)
    
    # Add some noise to face skin
    noise = np.random.randint(-15, 15, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Re-draw background to black
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), face_radius, 255, -1)
    img = cv2.bitwise_and(img, mask)

    # Coordinates
    left_eye_center = (17, 18)
    right_eye_center = (31, 18)
    eye_color = 40
    feature_thickness = 1
    
    # Draw Eyes based on emotion
    if emotion == "surprise":
        # Surprise: large open circles
        cv2.circle(img, left_eye_center, 3, eye_color, 1)
        cv2.circle(img, right_eye_center, 3, eye_color, 1)
    elif emotion == "fear":
        # Fear: medium open circles
        cv2.circle(img, left_eye_center, 2, eye_color, 1)
        cv2.circle(img, right_eye_center, 2, eye_color, 1)
    elif emotion == "disgust":
        # Disgust: squeezed eyes (horizontal lines or small dots)
        cv2.line(img, (15, 18), (19, 18), eye_color, feature_thickness)
        cv2.line(img, (29, 18), (33, 18), eye_color, feature_thickness)
    else:
        # Default eyes (small filled circles)
        cv2.circle(img, left_eye_center, 2, eye_color, -1)
        cv2.circle(img, right_eye_center, 2, eye_color, -1)
        
    # Draw Eyebrows
    eb_color = 30
    if emotion == "angry":
        # Slanted eyebrows downwards: \  /
        cv2.line(img, (14, 12), (20, 16), eb_color, feature_thickness)
        cv2.line(img, (34, 12), (28, 16), eb_color, feature_thickness)
    elif emotion == "sad":
        # Slanted eyebrows upwards: /  \
        cv2.line(img, (14, 15), (20, 12), eb_color, feature_thickness)
        cv2.line(img, (34, 15), (28, 12), eb_color, feature_thickness)
    elif emotion == "surprise" or emotion == "fear":
        # Raised, arched eyebrows
        cv2.ellipse(img, (17, 15), (4, 2), 0, 180, 360, eb_color, feature_thickness)
        cv2.ellipse(img, (31, 15), (4, 2), 0, 180, 360, eb_color, feature_thickness)
    else:
        # Neutral/Happy/Disgust: flat eyebrows
        cv2.line(img, (14, 13), (20, 13), eb_color, feature_thickness)
        cv2.line(img, (28, 13), (34, 13), eb_color, feature_thickness)
        
    # Draw Mouth
    mouth_color = 50
    if emotion == "happy":
        # Smile: curved up arc
        cv2.ellipse(img, (24, 26), (8, 8), 0, 0, 180, mouth_color, feature_thickness + 1)
    elif emotion == "sad" or emotion == "angry":
        # Frown: curved down arc
        cv2.ellipse(img, (24, 36), (8, 6), 0, 180, 360, mouth_color, feature_thickness + 1)
    elif emotion == "surprise":
        # Surprised mouth: wide open circle/oval
        cv2.circle(img, (24, 32), 5, mouth_color, -1)
    elif emotion == "fear":
        # Fear: moderately open oval
        cv2.ellipse(img, (24, 32), (6, 3), 0, 0, 360, mouth_color, -1)
    elif emotion == "disgust":
        # Disgust: crooked mouth
        cv2.line(img, (18, 30), (22, 33), mouth_color, feature_thickness)
        cv2.line(img, (22, 33), (26, 29), mouth_color, feature_thickness)
        cv2.line(img, (26, 29), (30, 31), mouth_color, feature_thickness)
    else:
        # Neutral: straight horizontal line
        cv2.line(img, (18, 31), (30, 31), mouth_color, feature_thickness + 1)
        
    # Add minor Gaussian blur to smooth pixels and make it look organic
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def generate_dataset(num_train_per_class: int = 150, num_val_per_class: int = 30):
    print("Generating synthetic FER-2013-formatted dataset...")
    
    for split, count in [("train", num_train_per_class), ("val", num_val_per_class)]:
        split_dir = DATA_DIR / split
        print(f"Creating split: {split} ({count} images per class)")
        
        for emotion in EMOTIONS:
            emotion_dir = split_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                face = create_synthetic_face(emotion)
                filename = f"{emotion}_{i:04d}.png"
                cv2.imwrite(str(emotion_dir / filename), face)
                
    print(f"Dataset successfully created in: {DATA_DIR.resolve()}")

if __name__ == "__main__":
    generate_dataset()

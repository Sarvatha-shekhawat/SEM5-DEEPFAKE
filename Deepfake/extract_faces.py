from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

# Global MTCNN instance for faster single-frame processing
global_mtcnn = MTCNN(keep_all=False)

def extract_faces_from_video(video_path, output_dir, max_frames=60):
    mtcnn = global_mtcnn
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        total_frames = 300
        fps = 30

    # Dynamically select frames to analyze based on video length
    # Target approx 2 frames per second for high accuracy
    duration = total_frames / fps
    target_frames = int(duration * 2)
    
    if target_frames > max_frames:
        target_frames = max_frames
    elif target_frames < 5:
        target_frames = min(5, total_frames)

    frame_interval = max(1, total_frames // target_frames)

    os.makedirs(output_dir, exist_ok=True)
    frames_extracted = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        if i % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            face = mtcnn(img)
            if face is not None:
                img_pil = to_pil(face)
                save_path = os.path.join(output_dir, f"frame{frames_extracted}.jpg")
                img_pil.save(save_path)
                frames_extracted += 1
                
            if frames_extracted >= target_frames:
                break

    cap.release()

def extract_face_from_image(img_pil: Image.Image):
    """Takes a PIL Image, extracts the face, and returns a cropped PIL Image or None."""
    face_tensor = global_mtcnn(img_pil)
    if face_tensor is not None:
        return to_pil(face_tensor)
    return None

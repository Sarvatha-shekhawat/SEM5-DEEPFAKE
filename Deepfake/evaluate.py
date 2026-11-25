import os

import torch
from PIL import Image
from torchvision import transforms


_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet normalisation — required for EfficientNet pretrained weights
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]
)


def _predict_from_tensors(model, batch: torch.Tensor):
    with torch.no_grad():
        outputs = model(batch)
        softmax = torch.nn.functional.softmax(outputs, dim=1)

        frame_fake_scores = [round(float(p) * 100, 2) for p in softmax[:, 1].tolist()]
        fake_prob_mean = softmax[:, 1].mean().item()
        fake_prob_max = softmax[:, 1].max().item()
        # Give more weight to the most suspicious frame
        fake_prob = 0.6 * fake_prob_mean + 0.4 * fake_prob_max
        real_prob = softmax[:, 0].mean().item()

    if fake_prob >= 0.42:
        predicted_class = 1
        confidence = fake_prob * 100
    else:
        predicted_class = 0
        confidence = real_prob * 100

    return predicted_class, round(confidence, 2), frame_fake_scores


def _measure_faces(pil_images):
    """Measurements on the same face crops the classifier just scored."""
    import cv2
    import numpy as np

    if not pil_images:
        return None

    grays = []
    for img in pil_images:
        small = cv2.resize(np.array(img.convert("RGB")), (128, 128))
        grays.append(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))

    diffs = []
    for prev, nxt in zip(grays, grays[1:]):
        diffs.append(float(cv2.absdiff(prev, nxt).mean()) / 255.0)

    lighting = []
    textures = []
    for gray in grays:
        _, width = gray.shape
        left = float(gray[:, : width // 2].mean())
        right = float(gray[:, width // 2 :].mean())
        lighting.append(abs(left - right) / 255.0)
        textures.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    jump = max(range(len(diffs)), key=lambda i: diffs[i]) if diffs else None
    return {
        "temporal": (sum(diffs) / len(diffs)) if diffs else None,
        "peak_change": diffs[jump] if jump is not None else None,
        "jump_from": (jump + 1) if jump is not None else None,
        "jump_to": (jump + 2) if jump is not None else None,
        "lighting": sum(lighting) / len(lighting),
        "texture": sum(textures) / len(textures),
        "face_count": len(grays),
    }


def _load_face_images(folder_path):
    face_images = []
    if not os.path.isdir(folder_path):
        return face_images
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                face_images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
    return face_images


def predict_video(model, folder_path, detailed=False):
    face_images = _load_face_images(folder_path)
    if not face_images:
        if detailed:
            return -1, 0.0, [], None
        return -1, 0.0

    inputs = torch.cat([_transform(img).unsqueeze(0) for img in face_images])
    predicted_class, confidence, frame_scores = _predict_from_tensors(model, inputs)
    if not detailed:
        return predicted_class, confidence
    return predicted_class, confidence, frame_scores, _measure_faces(face_images)


def predict_frame(model, image: Image.Image, detailed=False):
    face = image.convert("RGB")
    tensor = _transform(face).unsqueeze(0)
    predicted_class, confidence, frame_scores = _predict_from_tensors(model, tensor)
    if not detailed:
        return predicted_class, confidence
    return predicted_class, confidence, frame_scores, _measure_faces([face])


def evaluate_model(model, dataset):
    from torch.utils.data import DataLoader

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=16):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


def evaluate_model(model, dataset):
    from torch.utils.data import DataLoader

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=16):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

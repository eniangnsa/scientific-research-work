import numpy as np
import torch
from torchvision.models import resnet50
from torchvision import transforms

# Assuming video_frames is a list of frames from the video
def extract_features(frames):
    model = resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for frame in frames:
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            feature = model(input_batch)
        features.append(feature.squeeze().numpy())
    
    return np.array(features)

video_features = extract_features(video_frames)

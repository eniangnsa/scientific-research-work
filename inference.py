import torch

def detect_anomalies(model, video_features):
    model.eval()
    with torch.no_grad():
        src = torch.tensor(video_features).unsqueeze(0)
        tgt = torch.zeros_like(src)  # or provide target sequence if needed
        anomaly_pred, _, _ = model(src, tgt)
        anomaly_scores = F.softmax(anomaly_pred, dim=-1)[:, 1]  # Probability of being anomalous
        anomalies = anomaly_scores > threshold  # Define your threshold
        return anomalies

anomalies = detect_anomalies(model, video_features)

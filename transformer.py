import torch.nn as nn
import torch.nn.functional as F

class MultiTaskTransformer(nn.Module):
    def __init__(self, num_tasks, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(MultiTaskTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers)
        self.fc_anomaly = nn.Linear(d_model, 2)  # Binary classification (normal/anomalous)
        self.fc_object = nn.Linear(d_model, num_object_classes)  # Object detection
        self.fc_activity = nn.Linear(d_model, num_activity_classes)  # Activity recognition

    def forward(self, src, tgt):
        transformer_output = self.transformer(src, tgt)
        anomaly_pred = self.fc_anomaly(transformer_output)
        object_pred = self.fc_object(transformer_output)
        activity_pred = self.fc_activity(transformer_output)
        return anomaly_pred, object_pred, activity_pred

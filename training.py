from torch import nn

criterion_anomaly = nn.CrossEntropyLoss()
criterion_object = nn.CrossEntropyLoss()
criterion_activity = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in train_loader:
        src, tgt, anomaly_labels, object_labels, activity_labels = batch
        optimizer.zero_grad()
        
        anomaly_pred, object_pred, activity_pred = model(src, tgt)
        
        loss_anomaly = criterion_anomaly(anomaly_pred, anomaly_labels)
        loss_object = criterion_object(object_pred, object_labels)
        loss_activity = criterion_activity(activity_pred, activity_labels)
        
        loss = loss_anomaly + loss_object + loss_activity
        loss.backward()
        optimizer.step()

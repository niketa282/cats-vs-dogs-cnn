import torch
from model import CatsDogsCNN
from dataset import test_dataloader, train_dataloader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
model = CatsDogsCNN().to(device)
model.load_state_dict(torch.load("catsdogs_best_model.pth", map_location=device))

model.eval()

all_labels = []
all_predictions = []

# Test accurac

with torch.no_grad():
    for batch_number, (images, labels) in enumerate(tqdm(test_dataloader,
        desc="Testing")):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        predicted_classes = output.argmax(dim=1)
       
        all_labels.extend(
            labels.cpu().tolist()
        )
        
        all_predictions.extend(
            predicted_classes.cpu().tolist()
        )
        
        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )
        
        confusion = confusion_matrix(
            all_labels,
            all_predictions
        )

        precision = precision_score( # when the model says dog how often is it right?
            all_labels,
            all_predictions,
            average="binary"
        )

        recall = recall_score( # of all the dogs, how many did the model correctly identify?
            all_labels,
            all_predictions,
            average="binary"
       )
        
        f1 = f1_score(
            all_labels,
            all_predictions,
            average="binary"
     )
        
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Cat", "Dog"]
    )
)
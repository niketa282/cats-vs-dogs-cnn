import torch
from model import CatsDogsCNN
from dataset import test_dataloader, train_dataloader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
model = CatsDogsCNN().to(device)
model.load_state_dict(torch.load("catsdogs_model.pth", map_location=device))

model.eval()

# Training accuracy
correct_predictions_training = 0
total_predictions_training = 0

with torch.no_grad():
    for batch_number, (images, labels) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        predicted_classes = output.argmax(dim=1)
        correct_predictions_training += (predicted_classes == labels).sum().item()
        total_predictions_training += len(labels)
training_accuracy = 100 * correct_predictions_training / total_predictions_training

print(f"Final Training Accuracy: {training_accuracy:.2f}%")


# Test accuracy

correct_predictions_test = 0
total_predictions_test = 0

with torch.no_grad():
    for batch_number, (images, labels) in enumerate(tqdm(test_dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        predicted_classes = output.argmax(dim=1)
       # print(images.shape)  [batch size, 3, 128, 128]
       # print(output.shape)   [batch size, 2]
       # print(predicted_classes.shape)  [batch size]
       # print(f"Predicted: {predicted_classes}") prints the tensor
        correct_predictions_test += (predicted_classes == labels).sum().item()
        total_predictions_test += len(labels)
    test_accuracy = correct_predictions_test / total_predictions_test * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

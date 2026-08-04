import torch
from torch import optim
from dataset import train_dataloader, validation_dataloader
from model import CatsDogsCNN
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = CatsDogsCNN().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []

best_validation_accuracy = 0.0

for epoch in range(num_epochs):
 #-----------------------------------   
 # TRAINING
 #-------------------------------------   
    model.train()
    running_loss = 0.0
    correct_train_predictions = 0
    total_train_predictions = 0
    
    for batch_number, (images, labels) in enumerate(tqdm( train_dataloader,
        desc=f"Training epoch {epoch + 1}")):

        images = images.to(device)
        labels = labels.to(device)
       
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        predicted_classes = output.argmax(dim=1)
        correct_train_predictions += (predicted_classes == labels).sum().item()
        total_train_predictions += labels.size(0)
   
    # Calculate average loss for this epoch
    average_train_loss = running_loss / len(train_dataloader)
    train_accuracy = (
        correct_train_predictions
        / total_train_predictions
        * 100
    )

    # Store it for plotting
    train_losses.append(average_train_loss)
    train_accuracies.append(train_accuracy)

 
 #-----------------------------------   
 # VALIDATION
 #------------------------------------- 
 
    model.eval()
    
    running_validation_loss = 0.0
    correct_validation_predictions = 0
    total_validation_predictions = 0
    
    with torch.no_grad():

        for images, labels in tqdm(
            validation_dataloader,
            desc=f"Validation epoch {epoch + 1}"
        ):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_validation_loss += loss.item()
            predicted_classes = outputs.argmax(dim=1)
            
            correct_validation_predictions += (
                predicted_classes == labels
            ).sum().item()

            total_validation_predictions += labels.size(0)
            
        average_validation_loss = (
            running_validation_loss
          / len(validation_dataloader)
    )
        validation_accuracy = (
            correct_validation_predictions
            / total_validation_predictions
            * 100
    )
        validation_losses.append(
        average_validation_loss
    )

    validation_accuracies.append(
        validation_accuracy
    )
    
    print(
        f"\nEpoch {epoch + 1}/{num_epochs}\n"
        f"Training loss: {average_train_loss:.4f} | "
        f"Training accuracy: {train_accuracy:.2f}%\n"
        f"Validation loss: {average_validation_loss:.4f} | "
        f"Validation accuracy: {validation_accuracy:.2f}%"
    )
    
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            "catsdogs_best_model.pth"
        )

        print(
            "New best model saved! "
            f"Validation accuracy: "
            f"{best_validation_accuracy:.2f}%"
        )
        
epochs = range(1, num_epochs + 1)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training loss"
)

plt.plot(
    epochs,
    validation_losses,
    marker="o",
    label="Validation loss"
)

plt.xticks(epochs)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_plot_iridis.png")
plt.close()


plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Training accuracy"
)

plt.plot(
    epochs,
    validation_accuracies,
    marker="o",
    label="Validation accuracy"
)

plt.xticks(epochs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("accuracy_plot_iridis.png")
plt.close()
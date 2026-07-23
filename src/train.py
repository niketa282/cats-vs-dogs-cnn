import torch
from torch import optim
from dataset import train_dataloader
from model import CatsDogsCNN
import matplotlib.pyplot as plt
from tqdm import tqdm

model = CatsDogsCNN()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_number, (images, labels) in enumerate(tqdm(train_dataloader)):

        output = model(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
    # Calculate average loss for this epoch
    average_loss = running_loss / len(train_dataloader)

    # Store it for plotting
    train_losses.append(average_loss)

    # Print it
    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Average loss: {average_loss:.4f}"
    )    
 
epochs = range(1, num_epochs + 1)       
plt.plot(epochs, train_losses, marker="o")
plt.xticks(epochs)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Time")
plt.show()
plt.savefig("training_loss_plot.png")

torch.save(model.state_dict(), "catsdogs_model.pth")
print("Model saved successfully!")

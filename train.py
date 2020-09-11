import torch
import torch.nn as nn
from model import VGG11
from dataset import PlantSeedlingDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import copy


def train():
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    train_data = PlantSeedlingDataset(root_dir='train', transform=data_transforms)
    data_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG11(num_classes=train_data.num_classes).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loss, train_acc = [], []
    
    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0

    num_epochs = 80

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        running_train_loss = 0.0
        running_train_acc = 0
        for i, data in enumerate(data_loader):
            images, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            running_train_acc += torch.sum(preds==labels)

        print('Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(running_train_loss / len(train_data),
                                                                        torch.true_divide(running_train_acc, len(train_data))))

        train_loss.append(running_train_loss / len(train_data))
        train_acc.append(torch.true_divide(running_train_acc, len(train_data)))

        if running_train_acc > best_acc:
            best_acc = running_train_acc
            best_model_params = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_params)
        torch.save(model, 'VGG11_model.pth')

    plt.title("Loss Curve")
    plt.plot(range(num_epochs), train_loss, color='red', label="Training loss")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.savefig("loss_curve_cross_entropy.png")
    plt.show()
    
    plt.title("Accuracy Curve")
    plt.plot(range(num_epochs), train_acc, color='red', label="Training Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("Epochs")
    plt.savefig("accuracy_curve_cross_entropy.png")
    plt.show()


if __name__ == '__main__':
    train()

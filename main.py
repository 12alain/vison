from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.optim import Adam

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose,Resize
from torchvision.datasets import ImageFolder
#from utilsd import load_data
from data_tumeur import tumor
from model import Classifier


if __name__ == '__main__':
        
        
       
        image_size = (224, 224)

        
        transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor()])



        path    = 'brain_tumor_dataset'
        dataset = tumor(path,  transform=transform)
        
        print(dataset.classes)

        train_set,valid_set=torch.utils.data.random_split(dataset,[200,53])
        clf = Classifier()


        train_loader=DataLoader(train_set, batch_size=4, shuffle=True)
        valid_loader=DataLoader(valid_set, batch_size=4)

     
   
        opt=Adam(clf.parameters(),lr=1e-5)
        loss_fn=nn.CrossEntropyLoss()
        train_losses = []
        test_losses = []
        test_accuracies = []

        for epoch in range(10): 
            clf.train()
            correct = 0.0
            items = 0.0
            running_loss = 0.0

            for X, y in train_loader:
                X, y = X, y
                yhat = clf(X)
                loss = loss_fn(yhat, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                pred = torch.argmax(yhat, 1)
                correct += (y == pred).sum().item()
                items += y.size(0)
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # Évaluation sur les données de validation
            clf.eval()
            correct = 0.0
            items = 0.0
            test_loss = 0.0

            with torch.no_grad():
                for X, y in valid_loader:
                    X, y = X, y
                    yhat = clf(X)
                    loss = loss_fn(yhat, y)
                    pred = torch.argmax(yhat, 1)
                    correct += (y == pred).sum().item()
                    items += y.size(0)
                    test_loss += loss.item()

            test_losses.append(test_loss / len(valid_loader))
            test_accuracy = correct * 100 / items
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch + 1} - Train Loss: {train_loss}, Test Loss: {test_loss / len(valid_loader)}, Test Accuracy: {test_accuracy}%")
        

        torch.save(clf.state_dict(), './model/modele_train_tumeur(1).pth')
        # Tracer la courbe d'apprentissage
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), train_losses, label='Train Loss')
        plt.plot(range(1, 11), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Tracer la courbe de précision
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.show()
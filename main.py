import os.path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
import numpy as np
import matplotlib.colors as mcolors

from AutoEncoder import AutoEncoder

# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 30
NUM_CLASSES = 10
TRAINED_MODEL_PATH = 'autoencoder.pth'


def get_dataloaders_mnist(batch_size, num_workers=0, train_transforms=None, test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms, download=True)
    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)
    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    return train_loader, valid_dataset, test_loader


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn):
    loss = []
    for features, _ in data_loader:
        features = features.view(features.size(0), -1)
        logits = model(features)
        loss.append(loss_fn(logits, features).item())
    return np.mean(loss)


def train_autoencoder(num_epochs, model, optimizer,
                        train_loader, loss_fn=None,
                         logging_interval=100,
                         skip_epoch_stats=False,
                         save_model=None,
                         device='cpu'):
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    if loss_fn is None:
        loss_fn = F.mse_loss
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):
            # FORWARD AND BACK PROP
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))
        if not skip_epoch_stats:
            model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                    epoch + 1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    return log_dict


if __name__ == '__main__':
    # If there is a GPU available, use it
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    device = torch.device(DEVICE)

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size=BATCH_SIZE, num_workers=2)

    # Checking the dataset
    print('Training Set:\n')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break

    model = AutoEncoder()
    model.to(device)
    # If there is a trained model, load it
    if os.path.exists(TRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only=True))
        print("\n====================================================")
        print("Model loaded from:", TRAINED_MODEL_PATH)
        print("====================================================")
    else:
        print("Training the model")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        log_dict = train_autoencoder(num_epochs=NUM_EPOCHS, model=model,
                                     optimizer=optimizer,
                                     train_loader=train_loader,
                                     skip_epoch_stats=True,
                                     logging_interval=250,
                                     device=DEVICE)

        # Save the model
        torch.save(model.state_dict(), 'autoencoder.pth')

        # Plotting the training curves
        plt.plot(log_dict['train_loss_per_batch'], label='Training loss')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    # Testing the model
    print("Testing the model")
    for images, labels in test_loader:
        # images = images.view(images.size(0), -1)
        outputs = model(images.to(device))
        break

    images = images.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()

    n_images = 5
    plt.figure(figsize=(20, 4))
    for index in range(n_images):
        # Display original
        ax = plt.subplot(2, n_images, index + 1)
        plt.imshow(images[index].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n_images, index + 1 + n_images)
        plt.imshow(outputs[index].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

    # Visualizing the latent space
    for images, labels in test_loader:
        outputs = model.encoder(images.to(device))
        break

    outputs = outputs.cpu().detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(outputs[:, 0], outputs[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.grid()
    plt.show()



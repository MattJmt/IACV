from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
#from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau


def train_model(model, 
                train_loader, 
                val_loader, 
                num_epochs,
                best_metric, # "loss" or "accuracy"
                device, 
                save_path,
                lr_schedule_epochs: tuple = (3, 6),
                lr_schedule_gamma: float = 0.05):
                
    # initialise the best validation metric to ± infinity
    best_val_metric = float("inf") if best_metric == "loss" else float("-inf")
    best_val_epoch = 0

    # Create loss and accuracy logs for training and val
    train_loss_per_epoch, train_acc_per_epoch = [], []
    val_loss_per_epoch, val_acc_per_epoch = [], []

    model.to(device)  # select device
    """
    if lr_scheduler == "step":
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    elif lr_scheduler == "exp":
        scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_factor)
    """
    # Set initial optimizer parameters
    initial_lr = 0.01
    optimizer = optim.Adam(params=model.parameters(), lr=initial_lr)

    # Set criterion
    criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(0,num_epochs):
        # Adjust learning rate if specified in lr_schedule_epochs
        if epoch in lr_schedule_epochs:
            initial_lr *= lr_schedule_gamma
            optimizer = optim.Adam(params=model.parameters(), lr=initial_lr)

        # Training
        all_corrects, all_samples, total_loss = 0, 0, 0.0
        model.train()

        # Choose appropriate layers for optimization based on the epoch
        if epoch <= 5:
            layers_to_optimize = model.parameters()
        elif 5 < epoch <= 10:
            layers_to_optimize = model.layer1.parameters()
        else:
            layers_to_optimize = model.layer4.parameters()

        optimizer = optim.Adam(params=layers_to_optimize, lr=initial_lr)

        for images, labels in train_loader:
            # Move the images and labels to the device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  #predictinos
            corrects = torch.sum(preds == labels.data) #save correct ones
            loss = criterion(outputs, labels) #loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item() * len(labels) #update loss
            optimizer.step()    #update weights

            # Update the number of correct predictions and samples
            all_corrects += corrects
            all_samples += len(labels)

        # training loss, acc and save
        train_loss = total_loss / all_samples
        train_acc = float(all_corrects) / all_samples

        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(train_acc)

        # Validation
        all_corrects, all_samples, total_loss = 0, 0, 0.0
        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  #predictions
                loss = criterion(outputs, labels) #loss
                total_loss += loss.item() * len(labels)
                corrects = torch.sum(preds == labels.data) #save correct predictions

                # Update correct predictions and samples
                all_corrects += corrects
                all_samples += len(labels)

        # Val loss, acc, and save
        val_loss = total_loss / all_samples
        val_acc = float(all_corrects) / all_samples

        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_acc)

        # Update the best epoch
        if best_metric == "loss":   # minimising loss
            if val_loss < best_val_metric:
                best_val_metric, best_val_epoch = val_loss, epoch
                torch.save(model.state_dict(), save_path)
        else:       # maximising accuracy
            if val_acc > best_val_metric:
                best_val_metric, best_val_epoch = val_acc, epoch 
                torch.save(model.state_dict(), save_path)

         # Apply learning rate scheduling
        #scheduler.step(val_loss)  # For ReduceLROnPlateau, pass validation loss

        print(f"E{epoch} T: L|A {train_loss:.4f}|{train_acc:.4f}, V: L|A {val_loss:.4f}|{val_acc:.4f}, BE: {best_val_epoch}")

    # Plot the training curves
    plt.figure()
    plt.plot(np.array(train_loss_per_epoch))
    plt.plot(np.array(val_loss_per_epoch))
    plt.legend(['Training loss', 'Val loss'])
    plt.xlabel('Epoch')
    plt.show()

    plt.figure()
    plt.plot(np.array(train_acc_per_epoch))
    plt.plot(np.array(val_acc_per_epoch))
    plt.legend(['Training acc', 'Val acc'])
    plt.xlabel('Epoch')
    plt.show()
    plt.close()

    training_log = {
        "train_loss": train_loss_per_epoch,
        "train_acc": train_acc_per_epoch,
        "val_loss": val_loss_per_epoch,
        "val_acc": val_acc_per_epoch,
        "best_val_epoch": best_val_epoch,
    }

    return training_log

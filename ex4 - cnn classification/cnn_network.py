import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.net = nn.Sequential(
          nn.Conv2d(3, 6, 5, padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(4),
          nn.Conv2d(6, 48, 5, padding="same"),
          nn.MaxPool2d(4),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(432, 128),
          nn.Flatten(),
          nn.ReLU(),
          nn.Linear(128, 30),
          nn.Sigmoid(),
          nn.Linear(30, 6),
        )
        
        #self.conv1 = nn.Conv2d(3, 6, 5, padding=1)  # Increase output channels
        #self.conv2 = nn.Conv2d(6, 12, 3, padding=1)  # Additional convolutional layer
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv3 = nn.Conv2d(12, 64, 3, padding=1)  # Another convolutional layer
        #self.conv4 = nn.Conv2d(64, 128, 3, padding=1)  # Further convolutional layer
        #self.fc1 = nn.Linear(64 * 3 * 3, 32)  # Adjust input features for fully connected layers
        #self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, 6)  # Output layer for 10 classes (adjust as needed)

    def forward(self, x):
        #x = self.pool(self.conv1(x))

        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = self.fc1(x)
        #x = self.fc2(x)

        
        #x = self.pool(nn.functional.relu(self.conv1(x)))
        #x = self.pool(nn.functional.relu(self.conv2(x)))
        #x = self.pool(nn.functional.relu(self.conv3(x)))
        #x = self.pool(nn.functional.relu(self.conv4(x)))
        #x = x.view(-1, 64 * 3 * 3)  # Reshape tensor for fully connected layers
        #x = nn.functional.relu(self.fc1(x))
        #x = self.dropout(x)  # Apply dropout
        #x = nn.functional.relu(self.fc2(x))
        #x = self.fc3(x)
        
        return self.net(x)

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)

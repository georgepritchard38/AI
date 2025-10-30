import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np



def get_data_loader(training = True):
    """

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_set = datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set = datasets.FashionMNIST('./data',train=False,transform=custom_transform)

    data_set = None

    #determine which set to pass to loader based on function call
    if training:
        data_set = train_set
    else:
        data_set = test_set

    loader = torch.utils.data.DataLoader(data_set, batch_size = 32)

    return loader


def build_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        )
    
    return model



def build_deeper_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    deep_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
        )
    
    return deep_model



def train_model(model, train_loader, criterion, T):
    """

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.0115, momentum=0.9)

    #set model to train mode
    model.train()

    #train model
    for epoch in range(T):
        running_loss = 0.0
        accuracy = 0
        for batch, data in enumerate(train_loader, 0):
            inputs, labels = data

            #zero optimizer
            opt.zero_grad()

            #forward, backward, optimize
            outputs = model(inputs)
            
            #calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            accuracyPer = (accuracy / len(train_loader.dataset)) * 100

            #calculate average loss per batch
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            #calculate running loss of epoch
            running_loss += loss.item() / len(train_loader)

        #print training status for each epoch
        print("Train Epoch: ", epoch, " Accuracy: ", accuracy, "/", len(train_loader.dataset), 
              " (",round(accuracyPer,2),"%) ","Loss: ", round(running_loss, 3), sep="")





def evaluate_model(model, test_loader, criterion, show_loss = True):
    """

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #set model to evaluation mode
    model.eval()

    #train model
    with torch.no_grad():
        running_loss = 0.0
        accuracy = 0
        for batch, data in enumerate(test_loader, 0):
            inputs, labels = data

            #zero optimizer
            opt.zero_grad()

            #forward, backward, optimize
            outputs = model(inputs)
            
            #calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            accuracyPer = (accuracy / len(test_loader.dataset)) * 100

            #calculate average loss per batch
            loss = criterion(outputs, labels)
            opt.step()

            #calculate running loss
            running_loss += loss.item() / len(test_loader)

        #print evaluation results
        if show_loss:
            print("Average loss:", round(running_loss, 4))
        print("Accuracy: ",round(accuracyPer,2), "%", sep="")



def predict_label(model, test_images, index):
    """

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    #maps stored labels(indices 0-9) to known classes
    class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

    #calculate top 3 most likely labels
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)
    top3 = torch.topk(prob, k=3, sorted=True)

    #convert embedded tensors to cleaner lists
    values = (top3[0]).tolist()[0]
    labels = top3[1].tolist()[0]

    for index in range(len(values)):
        print(class_names[labels[index]], ": ", round((values[index] * 100), 2), "%", sep="")





if __name__ == '__main__':

    #train and evaluate shallower model
    print("--Shallow Model--")
    loader = get_data_loader()
    model = build_model()
    print("Train:")
    train_model(model, loader, 0.5, 10)
    print("Test:")
    evaluate_model(model, loader, 0.5, True)
    print()

    print("--Deep Model--")
    #train and evaluate deeper model
    deep_model = build_deeper_model()
    print("Train:")
    train_model(deep_model, loader, 0.5, 10)
    print("Test:")
    evaluate_model(deep_model, loader, 0.5, True)
    print()

    #predict labele of first output
    print("Predict first image label:")
    test_images = next(iter(loader))[0]
    predict_label(deep_model, test_images, 1)

    criterion = nn.CrossEntropyLoss()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    if(training == True): #Train set if true
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    return(loader)

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
        nn.Linear(64,10)
    )
    return(model)




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
    model.train()
    for epoch in range(T):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        runningLoss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            totalForLoss = labels.size(-1) #Gets the size of out dataset per epoch for loss
            runningLoss += loss.item() #Gets loss in the epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader:
                images, labels=data
                outputs=model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(-1)
                correct += (predicted == labels).sum().item()
        print(f'Train Epoch: {epoch}  Accuracy: {correct}/{total}({(correct/total)*100:.2f}%) Loss: {runningLoss/(totalForLoss*64):.3f}')
       


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    counter = 0
    runningLoss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels=data
            outputs=model(images)

            loss = criterion(outputs,labels)
            runningLoss += loss.item()/100
            counter+=1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
        if(show_loss == True):
            print(f'Average loss: {(runningLoss/counter):.4}')
        print(f'Accuracy: {(correct/total)*100:.2f}%')

    


def predict_label(model, test_images, index):
    """

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    #print(type(test_images))

    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']
    prob = F.softmax(model(test_images), dim = index)
    listed = prob[index].tolist()
    listed.sort(reverse=True)
    i1 = (prob[index].tolist()).index(listed[0])
    i2 = (prob[index].tolist()).index(listed[1])
    i3 = (prob[index].tolist()).index(listed[2])

    print(f'{class_names[i1]}: {(prob[index][i1])*100:.2f}%')
    print(f'{class_names[i2]}: {(prob[index][i2])*100:.2f}%')
    print(f'{class_names[i3]}: {(prob[index][i3])*100:.2f}%')


if __name__ == '__main__':
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    criterion = nn.CrossEntropyLoss()

    model = build_model()

    

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion)

    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)

    

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable

### data prepation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])
data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)
                           
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True,
                                                num_workers=0)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = True,
                                               num_workers=0)
### number of train                                               
print(len(data_train))

### show the part of images
images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean

print([labels[i] for i in range(64)])
plt.imshow(img)

### construct the cnn model
class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                         torch.nn.Tanh(),
                                         torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                         torch.nn.Tanh(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,30),
                                         torch.nn.Tanh(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(30, 10))
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x
    model = Model()
    print(model)

### loss and activities function
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

### trainning
n_epochs = 5
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format((epoch+1), n_epochs))
    print("-"*10)
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        y_onehot = torch.zeros([len(y_train), 10])
        y_onehot.scatter_(1, torch.reshape(y_train, (len(y_train),1)), 1)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_onehot)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    print("Train Loss is:{:.8f}, Train Accuracy is:{:.4f}%".format(torch.div(running_loss,len(data_train)),
                                                            torch.div(float(100*running_correct), len(data_train))) )  
### testing
#model.eval()
testing_loss = 0.0
testing_correct = 0
    #input("")
for data in data_loader_test:
    X_test, y_test = data
    X_test, y_test = Variable(X_test), Variable(y_test)
    y_testonehot = torch.zeros([len(y_test), 10])
    y_testonehot.scatter_(1, torch.reshape(y_test, (len(y_test),1)), 1)
    outputs = model(X_test)
    _, pred = torch.max(outputs.data, 1)
    testloss = cost(outputs, y_testonehot)
    testing_loss += testloss.item()
    testing_correct += torch.sum(pred == y_test.data)
    
print("Test Loss is:{:.8f}, Test Accuracy is:{:.4f}%".format(torch.div(testing_loss,len(data_test)),
                                                        torch.div(float(100*testing_correct), len(data_test))) )  
    
    
#torch.save(model.state_dict(), "model_parameter.pkl")

### test the input image
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                          batch_size = 4,
                                          shuffle = True)
X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_,pred = torch.max(pred, 1)

print("Predict Label is:", [ i for i in pred.data])
print("Real Label is:",[i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
plt.imshow(img)

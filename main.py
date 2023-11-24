import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline



if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    # print (x)
# else:
    # print ("MPS device not found.")

data_raw = pd.read_csv('SignLanguageConversion/data/sign_mnist_train.csv', sep=",")
test_data_raw = pd.read_csv('SignLanguageConversion/data/sign_mnist_test.csv', sep=",")

labels = data_raw['label']
data_raw.drop('label', axis=1, inplace=True)
labels_test = test_data_raw['label']
test_data_raw.drop('label', axis=1, inplace=True)

data = data_raw.values
labels = labels.values
test_data = test_data_raw.values
labels_test = labels_test.values


plt.figure(figsize=(10, 8))

pixels = data[10].reshape(28, 28)
plt.subplot(221)
sns.heatmap(data=pixels)

pixels = data[11].reshape(28, 28)
plt.subplot(222)
sns.heatmap(data=pixels)

pixels = data[20].reshape(28, 28)
plt.subplot(223)
sns.heatmap(data=pixels)

pixels = data[32].reshape(28, 28)
plt.subplot(224)
sns.heatmap(data=pixels)

choices = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:"f", 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n',
        14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}


def reshape_to_2d(data, dim):
    reshaped = []
    for i in data:
        reshaped.append(i.reshape(1, dim, dim))

    return np.array(reshaped)

data = reshape_to_2d(data, 28)

x = torch.FloatTensor(data)
y = torch.LongTensor(labels.tolist())

test_labels = torch.LongTensor(labels_test.tolist())

test_data_formated = reshape_to_2d(test_data, 28)
test_data_formated = torch.FloatTensor(test_data_formated)

epochs = 250
batch_size = 100
learning_rate = 0.001


class Network(nn.Module): 
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 10,kernel_size= 3)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(in_channels=10,out_channels= 20,kernel_size= 3)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3) 
        self.dropout1 = nn.Dropout2d()
        
        self.fc3 = nn.Linear(30 * 3 * 3, 270) 
        self.fc4 = nn.Linear(270, 26) 
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
                
        x = x.view(-1, 30 * 3 * 3) 
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return self.softmax(x)
    

    def test(self, predictions, labels):
        
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        print("Correct predictions: %5d / %5d (%5f)" % (correct, len(predictions), acc))
        
    
    def evaluate(self, predictions, labels):
                
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        return(acc)

torch.manual_seed(4)

model = Network()

lossFunction = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.7)
loss_func = nn.CrossEntropyLoss()

def train():
    loss_log = []
    acc_log = []

    for e in range(epochs):
        model.train()
        for i in range(0, x.shape[0], batch_size):
            x_mini = x[i:i + batch_size] 
            y_mini = y[i:i + batch_size] 
            
            optimizer.zero_grad()
            net_out = model(Variable(x_mini))
            
            loss = loss_func(net_out, Variable(y_mini))
            loss.backward()
            optimizer.step()
            
            if i % 1000 == 0:
                #pred = net(Variable(test_data_formated))
                loss_log.append(loss.item())
                acc_log.append(model.evaluate(torch.max(model(Variable(test_data_formated[:500])).data, 1)[1], 
                                            test_labels[:500]))
            
        print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))

def model_save():
     torch.save(model.state_dict(), # only saving the state_dict() only saves the models learned parameters
            "SignLanguageConversion/models.pth")

    # from pathlib import Path

    # 1. Create models directory 
    # MODEL_PATH = Path("PyTorch/SignLanguageConversion/models")
    # MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path 
    # MODEL_NAME = "01_pytorch_workflow_model_1.pth"
    # MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. Save the model state dict 
    # print(f"Saving model to: {MODEL_SAVE_PATH}")
    # torch.save(model.state_dict(), # only saving the state_dict() only saves the models learned parameters
    #         "PyTorch/SignLanguageConversion/models.pth")

# model1 = Network()


# model1.load_state_dict(torch.load("PyTorch/SignLanguageConversion/models.pth"))
# model1.eval()

# # Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
# # model.to(device)

# print(f"Loaded model:\n{model1}")
# print(f"Model on device:\n{next(model1.parameters()).device}")


# plt.figure(figsize=(10, 8))

# sample = 30
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(221)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))

# sample = 42
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(222)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))

# sample = 100
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(223)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))

# sample = 79
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(224)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))


# from PIL import Image
# import torchvision.transforms as transforms

# Read image

# def predict():
#     img = Image.open('g4g.png')
#     transformations = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                             transforms.Resize(size=50),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize([0.5],[0.5])])

#     image_tensor = transformations(img)[:3,:,:].unsqueeze(0)
#     pixels = image_tensor.reshape(28, 28)


if __name__ == "__main__":
    train()
    model_save()
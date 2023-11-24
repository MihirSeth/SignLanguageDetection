import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from main import Network





# from pathlib import Path

# # 1. Create models directory 
# MODEL_PATH = Path("PyTorch/SignLanguageConversion/models")


# 2. Create model save path 
# MODEL_NAME = "01_pytorch_workflow_model_1.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

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

choices = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:"f", 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n',
        14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}


# Instantiate a fresh instance of LinearRegressionModelV2
model = Network()


# Load model state dict 
model.load_state_dict(torch.load("SignLanguageConversion/models.pth"))
model.eval()

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
# model.to(device)

print(f"Loaded model:\n{model}")
print(f"Model on device:\n{next(model.parameters()).device}")


# plt.figure(figsize=(10, 8))

# sample = 265
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(221)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))

# sample = 33
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(222)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))

# sample = 11
# pixels = test_data[sample].reshape(28, 28)
# plt.subplot(223)
# sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))

# # sample = 72
# print(test_data[sample].shape)
# pixels = test_data[sample].reshape(28, 28)
# print(test_data[sample].reshape(1, 28, 28).shape)

# # plt.subplot(224)
# # sns.heatmap(data=pixels)
# lab = labels_test[sample]
# test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
# test_var_sample = Variable(test_sample)
# net_out_sample = model(test_var_sample)

# print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
# print("Actual Label: {}".format(choices[lab]))


# predictions = model(Variable(test_data_formated))
# model.test(torch.max(predictions.data, 1)[1], test_labels)


# for i in range (1,2):
#     sample = i
#     print(test_data[sample].size)

#     pixels = test_data[sample].reshape(28, 28)
#     print(test_data[sample].reshape(1, 28, 28).shape)

#     plt.subplot(243)
#     sns.heatmap(data=pixels)
#     lab = labels_test[sample]
#     test_sample = torch.FloatTensor([test_data[sample].reshape(1, 28, 28).tolist()])
#     test_var_sample = Variable(test_sample)
#     net_out_sample = model(test_var_sample)

  

#     print("Prediction: {}".format(choices[torch.max(net_out_sample.data, 1)[1].numpy()[0]]))
#     print("Actual Label: {}".format(choices[lab]))

#     probs, label = torch.topk(net_out_sample, 25)
#     probs = torch.nn.functional.softmax(probs, 1)
#     print(float(probs[0,0]))


# from PIL import Image
# import torchvision.transforms as transforms
import cv2


def predict():

    # image = cv2.imread('PyTorch/SignLanguageConversion/test.png', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    # image = cv2.resize(image, (28, 28))  # Resize the image to the model's input size
    # image = image / 255.0  # Normalize pixel values to [0, 1]

    #     # Convert the image to a PyTorch tensor
    # image_tensor = torch.FloatTensor(image).view(1, 28, 28)  # Assuming your model takes a single grayscale image

    #     # Use the model for prediction
    # with torch.no_grad():
    #         output = model(image_tensor)

    #     # Map the model's output to human-readable labels
    # predicted_class = torch.argmax(output).item()
    # predicted_label = choices[predicted_class]  # choices is your label mapping dictionary

    # return predicted_label

    image = cv2.imread('SignLanguageConversion/test/test6.png')
    
    # Resize the image to the input size expected by the model
    # print('test: ',image.shape)

    res = cv2.resize(image, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
    # res = res[:,:,0]
    # res = res/255

    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # print('test: ',res.shape)

    # print('hello: ', res.shape)

    

    # normalized_image = gray_image/ 255

    # res1 = torch.from_numpy(res1)
    res1 = torch.FloatTensor([gray_image.reshape(1, 28, 28).tolist()])
    print('test: ',res1.shape)



    # pixels = res.reshape(28, 28)
    # test_sample = torch.FloatTensor([res1.type])
    test_var_sample = Variable(res1)

    # out = model(res1)
    # probs, label = torch.topk(out, 25)
    # probs = torch.nn.functional.softmax(probs, 1)


    # test_sample = torch.FloatTensor([res1.reshape(1, 28, 28).tolist()])
    # test_var_sample = Variable(test_sample)
    # net_out_sample = model(test_var_sample)

    # Set the model to evaluation mode
    
    # Make a prediction
    # with torch.no_grad():
    prediction = model(test_var_sample)
    
    # Get the class index with the highest probability
    predicted_class = prediction.argmax().item()
    
    # Get the class name based on the class index
    class_name = choices[predicted_class]
    class_name = choices[torch.max(prediction, 1)[1].numpy()[0]]

    probs, label = torch.topk(prediction, 25)
    probs = torch.nn.functional.softmax(probs, 1)

    pred = prediction.max(1, keepdim=True)[1]

    
    return float(probs[0,0]), class_name

    # if float(probs[0,0]) < 0.4:
    #     texto_mostrar = 'Sign not detected'
    #     return texto_mostrar
    # else:
    #     texto_mostrar = choices[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])) + '%'
    #     return texto_mostrar



print(predict())





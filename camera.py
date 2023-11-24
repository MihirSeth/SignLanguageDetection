import cv2
import mediapipe as mp
from torch.autograd import Variable

from test import predict
from main import Network
import torch

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

choices = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:"f", 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n',
        14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}


# Instantiate a fresh instance of LinearRegressionModelV2
model = Network()


# Load model state dict 
model.load_state_dict(torch.load("SignLanguageConversion/models.pth"))
model.eval()

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # max_x = max(max_x, cx) - 5
                # max_y = max(max_y, cy) - 10
                # min_x = min(min_x, cx) + 10
                # min_y = min(min_y, cy) + 10

                min_x = min(min_x, cx) - 10
                min_y = min(min_y, cy) - 15
                max_x = max(max_x, cx) + 10
                max_y = max(max_y, cy) + 10

                if id == 20:
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            # Draw the bounding box
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            
            # frame = image[min_y:max_y, min_x:max_x]

            # image = cv2.imread('SignLanguageConversion/test/test37.png')
            


            res = cv2.resize(image, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)


            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            
            res1 = torch.FloatTensor([res.reshape(1, 28, 28).tolist()])



            test_var_sample = Variable(res1)

            prediction = model(test_var_sample)
            
            # Get the class index with the highest probability
            predicted_class = prediction.argmax().item()
            
            # Get the class name based on the class index
            class_name = choices[predicted_class]
            class_name = choices[torch.max(prediction, 1)[1].numpy()[0]]

            probs, label = torch.topk(prediction, 25)
            probs = torch.nn.functional.softmax(probs, 1)

            pred = prediction.max(1, keepdim=True)[1]

            
            print(float(probs[0,0]), class_name)


    cv2.imshow("Output", image)
    cv2.waitKey(1)














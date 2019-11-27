'''
@Author: haoMax
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293
@Date: 2019-11-21 10:07:13
@LastEditTime: 2019-11-27 14:35:26
@LastEditors: haoMax
@Description: 
'''
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
id2code={0: (64, 128, 64),
 1: (128, 128, 64),
 2: (0, 128, 192),
 3: (128, 0, 0),
 4: (0, 128, 64),
 5: (64, 0, 128),
 6: (64, 0, 192),
 7: (192, 128, 64),
 8: (192, 192, 128),
 9: (64, 64, 128),
 10: (128, 0, 192),
 11: (192, 0, 64),
 12: (192, 0, 128),
 13: (192, 0, 192),
 14: (128, 64, 64),
 15: (64, 192, 128),
 16: (64, 64, 0),
 17: (128, 64, 128),
 18: (128, 128, 192),
 19: (0, 0, 192),
 20: (192, 128, 128),
 21: (128, 128, 128),
 22: (64, 128, 192),
 23: (0, 0, 64),
 24: (0, 64, 64),
 25: (192, 64, 128),
 26: (128, 128, 0),
 27: (192, 128, 192),
 28: (64, 0, 64),
 29: (192, 192, 0),
 30: (0, 0, 0),
 31: (64, 192, 0)}
def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)
model = load_model('model.h5')
PIXEL = 512
X = []
for info in os.listdir(r'./images/validation/'):
    A = cv2.imread("./images/validation/" + info)
    A=cv2.resize(A,(PIXEL,PIXEL))
    X.append(A)
    # i += 1

X = np.array(X)

Y = model.predict(X)

# Y=Y.astype(int)
# print(np.unique(Y))
for i in range(len(X)):
    # print(np.unique(Y[i]))
    print(Y[i])
   #
    # print(Y[i].shape)
    # Y[i]=Y[i]*50
    print(np.unique(Y[i]))
    name=str(i)+'.png'

    # cv2.imwrite(name,Y[i][0]*50)
    # print()
    print(Y[i].shape)
    cv2.imshow("a",onehot_to_rgb(Y[i]))
    cv2.imshow("A",X[i])
    cv2.waitKey(0)


# groudtruth = []
# for info in os.listdir(r'G:\\haihan\\Segmentation\\data\\test_groudtruth'):
#     A = cv2.imread("data\\test_groudtruth\\" + info)
#     groudtruth.append(A)
# groudtruth = np.array(groudtruth)

# a = range(10)
# n = np.random.choice(a)
# cv2.imwrite('prediction.png',Y[n])
# cv2.imwrite('groudtruth.png',groudtruth[n])
# fig, axs = plt.subplots(1, 3)
# # cnt = 1
# # for j in range(1):
# axs[0].imshow(np.abs(X[n]))
# axs[0].axis('off')
# axs[1].imshow(np.abs(Y[n]))
# axs[1].axis('off')
# axs[2].imshow(np.abs(groudtruth[n]))
# axs[2].axis('off')
#     # cnt += 1
# fig.savefig("imagestest.png")
# plt.close()


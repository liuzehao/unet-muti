import cv2
import os
import numpy as np
train='./annotations/training'
for i in os.listdir(train):
    train_path=os.path.join(train,i)
    im=cv2.imread(train_path)
    im=im*255
    cv2.imshow("a",im)
    cv2.waitKey(0)



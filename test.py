from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
model = load_model('V1_828.h5')
PIXEL = 256
X = []
for info in os.listdir(r'./images/training/'):
    A = cv2.imread("./images/training/" + info)
    A=cv2.resize(A,(PIXEL,PIXEL))
    X.append(A)
    # i += 1

X = np.array(X)
print(X.shape)
Y = model.predict(X)*50
print(np.unique(Y))
cv2.imshow("a",Y[0])

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


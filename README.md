@[toc]
# unet-muti
多分类unet,baseline
https://blog.csdn.net/liu506039293/article/details/103234618


## 前言

最近研究了三种分割算法，deeplab-v3-plus，FCN，还有Une。FCN是分割网络的开山之作，可以用来学习，deeplab-v3-plus速度比较慢，精度更高，代码改起来比较复杂。落地的话首选还是UNET，相比较与目标检测的网络，代码简单到爆炸，也推荐作为深度学习的入门网络。
## 网络结构
可以看到整个网络结构是一个U型的结构，前面部分通过pooling进行下采样，后面部分通过反卷积上采样。中间通过concatenate进行拼接。
```python
inputs = Input((PIXEL, PIXEL, 3))
s = Lambda(lambda x: x / 255) (inputs)
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(s)
pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  # 16

conv2 = BatchNormalization(momentum=0.99)(pool1)
conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
conv2 = BatchNormalization(momentum=0.99)(conv2)
conv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
conv2 = Dropout(0.02)(conv2)
pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # 8

conv3 = BatchNormalization(momentum=0.99)(pool2)
conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
conv3 = BatchNormalization(momentum=0.99)(conv3)
conv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
conv3 = Dropout(0.02)(conv3)
pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 4

conv4 = BatchNormalization(momentum=0.99)(pool3)
conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
conv4 = BatchNormalization(momentum=0.99)(conv4)
conv4 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
conv4 = Dropout(0.02)(conv4)
pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

conv5 = BatchNormalization(momentum=0.99)(pool4)
conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
conv5 = BatchNormalization(momentum=0.99)(conv5)
conv5 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
conv5 = Dropout(0.02)(conv5)
pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)
# conv5 = Conv2D(35, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
# drop4 = Dropout(0.02)(conv5)
pool4 = AveragePooling2D(pool_size=(2, 2))(pool3)  # 2
pool5 = AveragePooling2D(pool_size=(2, 2))(pool4)  # 1

conv6 = BatchNormalization(momentum=0.99)(pool5)
conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
up7 = (UpSampling2D(size=(2, 2))(conv7))  # 2
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
merge7 = concatenate([pool4, conv7], axis=3)

conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
up8 = (UpSampling2D(size=(2, 2))(conv8))  # 4
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
merge8 = concatenate([pool3, conv8], axis=3)

conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
up9 = (UpSampling2D(size=(2, 2))(conv9))  # 8
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
merge9 = concatenate([pool2, conv9], axis=3)

conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
up10 = (UpSampling2D(size=(2, 2))(conv10))  # 16
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)

conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
up11 = (UpSampling2D(size=(2, 2))(conv11))  # 32
conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up11)

# conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv12)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191125111000416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdTUwNjAzOTI5Mw==,size_16,color_FFFFFF,t_70)
## 关于多分类
网络上一般的博客就给了这样的单分类网络，我同时把我成功修改的多分类也给到读者。其实自己修改也很简单，有几个步骤和坑我来说明一下。

 1. 最后一层的输出：Logistic函数（sigmoid函数）只适用于二分类改成relu函数。
 2. loss函数：同样的，由binary_crossentropy改成mse
 3. 最最关键的原因，我觉得也是很多博主没成功改成多分类的原因：数据集label有问题！
 如果你也是采用labelme标注的话，一定会在json转换的时候发现如下问题：
 3.1 一次只能转换一个
 3.2 多个标签情况下，图片标签失配
 3.3 转换为独热标签错误
 请务必使用我提供的转换脚本（目前网上的处理我，还没看见一个对的），大坑！
https://github.com/liuzehao/FCN-tools/blob/master/json_to_dataset.py
使用方法：
```
 python json_to_dataset.py ./你的json路径
```
注意在使用的时候要修改为你的分类
```
NAME_LABEL_MAP = {
    '_background_': 0,
    "cat": 1,
    "dog": 2,
}
 
LABEL_NAME_MAP = ['0: _background_',
                  '1: cat',
                  '2: dog']
```
## 文件结构
Segmentation_training.py是多分类
Segmentation_training_one 是单分类
test.py是前向传播
tonoone.py是可视化label文件
注意：每个batch都是随机采样，所以需要命名方式如：train-029，label和img名字相同。或者改一下遍历文件的地方。
## 最后想说的
1.如果你也是刚入分割的坑的话，我觉得不用做我之前那些无谓的尝试了，一般问题直接上Unet吧，复杂问题Unet也是最具有扩展能力的。
 2.pooling下采样其实可以改成卷积，pooling会导致梯度递减
 3.关于分割。虽然分割代码简单，不要以为可以替代目标检测。原因如下：
 如果目标之间相似度比较高，请用目标检测的算法。如果说相似程度高，且不适合用回归框表示，可以用点的目标检测方法。血的教训！！


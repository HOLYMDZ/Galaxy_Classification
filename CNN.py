from skimage import io,transform
import os
import tensorflow as tf
import numpy as np
import time
from astropy.io import fits
from matplotlib import pyplot as plt
#输入图片大小
w = 68
h = 68
c = 3
#切割图片
def extract_thumb(x = 212,y = 212,size = 207):
    if size % 2 == 0:
        size = size + 1
    up_x = int(x - size / 2)
    dow_x = int(x + size / 2)
    up_y=int(y - size / 2)
    dow_y = int(y + size / 2)    
    return up_x, dow_x, up_y, dow_y

up_x, dow_x, up_y, dow_y = extract_thumb()

#读取图片
def read_img():
    data = fits.getdata('C:/Users/Dezheng Meng/Desktop/IAC_XXX_WINTER-master/morphology/Nair_Abraham_cat.fit', 1)
    idcat = data['dr7objid']
    ttype = data['TType']
    labels = ttype * 0 - 1
    #elliptical galaxies
    labels[np.where((ttype >= -5) & (ttype <= 0))] = 0
    #spiral galaxies
    labels[np.where((ttype > 0) & (ttype <= 10))] = 1
    imgs = []
    n = 1
    for name in idcat:
        print(str(n) + ':reading the images:' + str(name) + "_GZOO_.jpg")
        img = 'C:/Users/Dezheng Meng/Desktop/cutouts_jpeg_all/' + str(name) + "_GZOO_.jpg"
        imgs.append(img)
        n += 1
    return np.asarray(imgs,np.str),np.asarray(labels,np.int32)
    
data,label = read_img()


#打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]


#将所有数据分为训练集和测试集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]
#占位符
x = tf.placeholder(tf.float32, shape = [None, w, h, c], name = 'x')
y_ = tf.placeholder(tf.int32, shape = [None, 1], name = 'y_')
judge = tf.Variable(tf.fill([30, 1], 0.5))
#网络结构
conv1 = tf.layers.conv2d(
      inputs = x,
      filters = 32,
      kernel_size = [6, 6],
      padding = "same",
      activation = tf.nn.relu,
      kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))

dropout1 = tf.nn.dropout(conv1,0.5)

conv2 = tf.layers.conv2d(
      inputs = dropout1,
      filters = 64,
      kernel_size = [5, 5],
      padding = "same",
      activation = tf.nn.relu,
      kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))

pool1 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

dropout2 = tf.nn.dropout(pool1, 0.25)

conv3 = tf.layers.conv2d(
      inputs = dropout2,
      filters = 64,
      kernel_size = [5, 5],
      padding = "same",
      activation = tf.nn.relu,
      kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))

pool2 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2, 2], strides = 2)

dropout3 = tf.nn.dropout(pool2, 0.25)

conv4 = tf.layers.conv2d(
      inputs = dropout3,
      filters = 128,
      kernel_size = [2, 2],
      padding = "same",
      activation = tf.nn.relu,
      kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))
pool3 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2, 2], strides = 2)

dropout3 = tf.nn.dropout(pool3, 0.25)

conv5 = tf.layers.conv2d(
      inputs = dropout3,
      filters = 128,
      kernel_size = [3, 3],
      padding = "same",
      activation = tf.nn.relu,
      kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))

dropout4 = tf.nn.dropout(conv5, 0.25)

flat1 = tf.layers.flatten(dropout4)

dense1 = tf.layers.dense(inputs = flat1, 
                      units = 64, 
                      activation = tf.nn.relu,
                      kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))

dropout5 = tf.nn.dropout(dense1, 0.5)

dense2= tf.layers.dense(inputs = dropout5, 
                      units = 1, 
                      activation = tf.sigmoid,
                      kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
                      kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))

#优化器、损失函数、准确率
loss = tf.losses.log_loss(labels = y_, predictions = dense2)
train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.greater_equal(dense2, judge), tf.int32), y_)    
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#按批次取数据
def minibatches(inputs = None, targets = None, batch_size = None, shuffle = False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练集和测试集大小
n_epoch = 50
batch_size = 30

sess = tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
#绘图数据存储
train_loss_plt = []
train_acc_plt = []
val_loss_plt = []
val_acc_plt = []
#输出log用于tensorboard可视化
writer = tf.summary.FileWriter('C:/Users/Dezheng Meng/Desktop/logs/', sess.graph)

#开始训练
for epoch in range(n_epoch):
    start_time = time.time()
    print("epoch:" + str(epoch))
    #训练集
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle = True):
        x_train_b = []
        for i in x_train_a:
            x_train_b.append(transform.resize(io.imread(i)[up_x:dow_x, up_y:dow_y], (w, h)))
        _,err,ac = sess.run([train_op, loss,acc], feed_dict = {x: x_train_b, y_: np.reshape(y_train_a, (30, 1))})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: " + str(train_loss / n_batch))
    print("   train acc: " + str(train_acc / n_batch))
    train_loss_plt.append(train_loss / n_batch)
    train_acc_plt.append(train_acc / n_batch)
    #测试集
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle = False):
        x_val_b = []
        for i in x_val_a:
            x_val_b.append(transform.resize(io.imread(i)[up_x:dow_x, up_y:dow_y], (w, h)))
        err, ac = sess.run([loss, acc], feed_dict={x:x_val_b, y_: np.reshape(y_val_a, (30, 1))})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   validation loss: " + str(val_loss / n_batch))
    print("   validation acc: " + str(val_acc / n_batch))
    val_loss_plt.append(val_loss / n_batch)
    val_acc_plt.append(val_acc / n_batch)
    
    end_time = time.time()
    print("time:" + str(end_time-start_time) + "s")

sess.close()
#绘制曲线
plt.figure()
plt.plot(train_loss_plt)
plt.title("train_loss")
plt.figure()
plt.plot(train_acc_plt)
plt.title("train_acc")
plt.figure()
plt.plot(val_loss_plt)
plt.title("val_loss")
plt.figure()
plt.plot(val_acc_plt)
plt.title("val_acc")
plt.show()
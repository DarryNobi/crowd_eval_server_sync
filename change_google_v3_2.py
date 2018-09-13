# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def get_file_dir(file):
    image_dir = []
    label_dir = []
    r_csv = open(file)
    path_lists = list(csv.reader(r_csv))
    for i in path_lists:
        image_dir.append(i[0])
        label_dir.append(i[1])
    label_dir = [int(j) for j in label_dir]
    return image_dir,label_dir

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    
    input_queue = tf.train.slice_input_producer([image,label],shuffle=True)
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    #image = tf.image.resize_images(image,image_W,image_H)
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,capacity = capacity)
    
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch

def inception_v3_arg_scope(weight_decay=0.00004,    # L2正则的weight_decay
                           stddev=0.1,  # 标准差0.1
                           batch_norm_var_collection='moving_vars'):

  batch_norm_params = {  # 定义batch normalization参数字典
      'decay': 0.9997,  #衰减系数
      'epsilon': 0.001,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  # silm.arg_scope可以给函数自动赋予某些默认值
  # 会对[slim.conv2d, slim.fully_connected]这两个函数的参数自动赋值,
  # 使用slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)): # 对[slim.conv2d, slim.fully_connected]自动赋值

      # 嵌套一个slim.arg_scope对卷积层生成函数slim.conv2d的几个参数赋予默认值
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=trunc_normal(stddev), # 权重初始化器
        activation_fn=tf.nn.relu, # 激活函数
        normalizer_fn=slim.batch_norm, # 标准化器
        normalizer_params=batch_norm_params) as sc: # 标准化器的参数设置为前面定义的batch_norm_params
      return sc # 最后返回定义好的scope


# 生成V3网络的卷积部分
def inception_v3_base(inputs, scope=None):
  '''
  Args:
  inputs：输入的tensor
  scope：包含了函数默认参数的环境
  '''
  end_points = {} # 定义一个字典表保存某些关键节点供之后使用

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 对三个参数设置默认值
                        stride=1, padding='VALID'):

      #  因为使用了slim以及slim.arg_scope，我们一行代码就可以定义好一个卷积层
      #  相比AlexNet使用好几行代码定义一个卷积层，或是VGGNet中专门写一个函数定义卷积层，都更加方便
      #
      # 正式定义Inception V3的网络结构。首先是前面的非Inception Module的卷积层
      # slim.conv2d函数第一个参数为输入的tensor，第二个是输出的通道数，卷积核尺寸，步长stride，padding模式

      #一共有5个卷积层，2个池化层，实现了对图片数据的尺寸压缩，并对图片特征进行了抽象
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3],
                        stride=2, scope='Conv2d_1a_3x3')    # 149 x 149 x 32

      net = slim.conv2d(net, 32, [3, 3],
                        scope='Conv2d_2a_3x3')      # 147 x 147 x 32

      net = slim.max_pool2d(net,[3, 3], stride=2,
                            scope='MaxPool_3a_3x3')   # 73 x 73 x 32

      net = slim.conv2d(net, 64, [3, 3],
                        scope='Conv2d_4a_3x3')  # 71 x 71 x 64

      net = slim.max_pool2d(net, [3, 3], stride=2,
                            scope='MaxPool_5a_3x3') # 35 x 35 x 64


    '''
    三个连续的Inception模块组，三个Inception模块组中各自分别有多个Inception Module，这部分是Inception Module V3
    的精华所在。每个Inception模块组内部的几个Inception Mdoule结构非常相似，但是存在一些细节的不同
    '''
    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 设置所有模块组的默认参数
                        stride=1, padding='SAME'): # 将所有卷积层、最大池化、平均池化层步长都设置为1
      # 第一个模块组包含了三个结构类似的Inception Module
      '''    
--------------------------------------------------------    
      第一个Inception组   一共三个Inception模块
      '''
      with tf.variable_scope('Mixed_5b'): # 第一个Inception Module名称。Inception Module有四个分支
        # 第三个分支64通道1*1卷积,96的3*3,再接一个3*3
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')#35x35x96

        net = branch_2 # 将四个分支的输出合并在一起（第三个维度合并，即输出通道上合并）
      # 64+64+96+32 = 256
      # mixed_1: 35 x 35 x 256.
      '''
      因为这里所有层步长均为1，并且padding模式为SAME，所以图片尺寸不会缩小，但是通道数增加了。四个分支通道数之和
      64+64+96+32=256，最终输出的tensor的图片尺寸为35*35*256
      '''

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3') #17*17*288
        net = branch_2 # 输出尺寸定格在17 x 17 x 768
      # 384+96+288 = 768
      # mixed_3: 17 x 17 x 768.

      # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类
      end_points['Mixed_6e'] = net

      # 第三个inception模块组包含了三个inception module

      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1') #17*17*192
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3') #8*8*192
        net = branch_1 # 输出图片尺寸被缩小，通道数增加，tensor的总size在持续下降中
      # 320+192+768 = 1280
      # mixed_8: 8 x 8 x 1280.


      with tf.variable_scope('Mixed_7c'):
        
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = branch_3
      # 320+(384+384)+(384+384)+192 = 2048
      # mixed_10: 8 x 8 x 2048.

      return net, end_points
      #Inception V3网络的核心部分，即卷积层部分就完成了
      '''
      设计inception net的重要原则是图片尺寸不断缩小，inception模块组的目的都是将空间结构简化，同时将空间信息转化为
      高阶抽象的特征信息，即将空间维度转为通道的维度。降低了计算量。Inception Module是通过组合比较简单的特征
      抽象（分支1）、比较比较复杂的特征抽象（分支2和分支3）和一个简化结构的池化层（分支4），一共四种不同程度的
      特征抽象和变换来有选择地保留不同层次的高阶特征，这样最大程度地丰富网络的表达能力。
      '''


# V3最后部分
# 全局平均池化、Softmax和Auxiliary Logits
def inception_v3(inputs,
                 num_classes=1000, # 最后需要分类的数量（比赛数据集的种类数）
                 is_training=True, # 标志是否为训练过程，只有在训练时Batch normalization和Dropout才会启用
                 dropout_keep_prob=0.7, # 节点保留比率
                 prediction_fn=slim.softmax, # 最后用来分类的函数
                 spatial_squeeze=True, # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
                 reuse=None, # 是否对网络和Variable进行重复使用
                 scope='InceptionV3'): # 包含函数默认参数的环境

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], # 定义参数默认值
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], # 定义标志默认值
                        is_training=is_training):
      # 拿到最后一层的输出net和重要节点的字典表end_points
      net, end_points = inception_v3_base(inputs, scope=scope) # 用定义好的函数构筑整个网络的卷积部分

      # Auxiliary logits作为辅助分类的节点，对分类结果预测有很大帮助
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'): # 将卷积、最大池化、平均池化步长设置为1
        aux_logits = end_points['Mixed_6e'] # 通过end_points取到Mixed_6e
        # end_points['Mixed_6e']  --> 17x17x768
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
                    aux_logits, [5, 5], stride=3, padding='VALID',
                    scope='AvgPool_1a_5x5') #5x5x768

          aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                   scope='Conv2d_1b_1x1') #5x5x128

          # Shape of feature map before the final layer.
          aux_logits = slim.conv2d(
              aux_logits, 768, [5, 5],
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv2d_2a_5x5')  #1x1x768

          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')  # 1*1*1000

          if spatial_squeeze: # tf.squeeze消除tensor中前两个为1的维度。
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits # 最后将辅助分类节点的输出aux_logits储存到字典表end_points中

      # 处理正常的分类预测逻辑
      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        # net --> 8x8x2048
        net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                              scope='AvgPool_1a_8x8') #1x1x2048

        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net

        # 激活函数和规范化函数设为空
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1') # 1x1x1000
        if spatial_squeeze: # tf.squeeze去除输出tensor中维度为1的节点
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions') # Softmax对结果进行分类预测
  return logits, end_points # 最后返回logits和包含辅助节点的end_points

def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits,labels = labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name = 'loss')
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #reg_constant = 0.01  # Choose an appropriate one.
        #loss = loss1 + reg_constant * sum(reg_losses)
        #loss = loss1 + reg_constant * tf.reduce_mean(reg_losses,name = 'loss2')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def trainning(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy



N_CLASSES = 4
IMG_W = 299
IMG_H = 299
BATCH_SIZE = 64
CAPACITY = 107*64
MAX_STEP = 8000
learning_rate = 0.0001

def run_training():
    train_dir = 'train.csv'
    logs_train_dir = 'bus/logs'
    train,train_label = get_file_dir(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
    #with slim.arg_scope(inception_v3_arg_scope()):
    train_logits,end_points =inception_v3(train_batch,num_classes=4)
    train_loss = losses(train_logits,train_label_batch)
    train_op = trainning(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            if step %  50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
                
            if step % 1000 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()
'''
run_training() 
'''
def get_one_image(img_dir):
     image = Image.open(img_dir)
     #plt.imshow(image)
     image = image.resize([299, 299])
     image_arr = np.array(image)
     return image_arr

def test(test_file):
    log_dir = 'bus/logs/'
    image_arr = get_one_image(test_file)
    fo = open("result.txt","w")
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,299, 299, 3])
        #print(image.shape)
        #with slim.arg_scope(inception_v3_arg_scope()):
        test_logits,end_points =inception_v3(image,num_classes=4,is_training=False)
        logits = tf.nn.softmax(test_logits)
        x = tf.placeholder(tf.float32,shape = [299,299,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})          
            print(prediction)
            max_index = np.argmax(prediction) 
            fo.write(str(prediction))
            fo.close()
            print(max_index)
            name = test_file.split('/')[-2]
            name = np.int(name)
            print(name)
            if max_index==name:
                return 1
            else:
                return 0


test_file_dir,label_dir= get_file_dir("test.csv")

def run(dir):
    num = 0
    for i in np.arange(100):
        num += test(dir[i+350*2])
        print(i)
    return num,num/100
correct_num,correct_rate = run(test_file_dir)
print (correct_num,correct_rate)








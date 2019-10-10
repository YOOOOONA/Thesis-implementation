# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:33:44 2019

@author: 융
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

#옵션 값 사전 정의
n_input = 100*100
n_class = 3
n_noise = 128
total_epoch = 10
batch_size = 1461
learning_rate = 0.0002
n_hidden = 256

#실제이미지, 라벨 파싱
train_list, test_list = [], []
with open('C:/Users/pc/Desktop/train.txt') as f:
    for line in f:
        tmp = line.strip().split()
        #[0]은 jpg이름
        #[1]은 과일의 인덱스값, 오렌지=2, 바나나=1,사과=0
        train_list.append([tmp[0],tmp[1]])
with open('C:/Users/pc/Desktop/test.txt') as f:
    for line in f:
        tmp = line.strip().split()
        test_list.append([tmp[0],tmp[1]])

#이미지 배열화
def readimg(path):
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #img=plt.imread(path, cmap="gray")
    img.show()
    img=np.reshape(img,[-1,10000])#이미지를 1차원배열화

def batch(train_list,batch_size):
    img, label, path =[],[],[]
    for i in range(batch_size):
        img.append(readimg(train_list[0][0]))
        label_list=[0 for _ in range(n_class)]
        label_list[int(train_list[0][1])]=int(train_list[0][1])
        label.append(label_list)
        
        path.append(train_list.pop(0))
    return img, label

#플레이스 홀더 설정
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])

#생성자 신경망
def generator(noise, labels):
    with tf.variable_scope('generator'):
        #noise값 을 껍데기를 벗겨서 각각의 값에 labels 정보를 추가하고 껍데기로 다시 싸기
        inputs = tf.concat([noise, labels],1)
        
        hidden = tf.layers.dense(inputs, n_hidden,#인풋값,아웃풋개수
                                 activation = tf.nn.relu)#만개넣어서 256개 로 출력
        output = tf.layers.dense(hidden,n_input,#인풋값,아웃풋갯수
                                 activation = tf.nn.sigmoid)#256에서 만개로
        return output
#구분자 신경망
def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:    
        
    
    
    
    
    
    
    
    
    


    
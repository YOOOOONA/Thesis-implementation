# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:14:28 2019

@author: 융
"""
import tensorflow as tf
t1=[[29, 29,  34]]
t2=[[29, 29, 17]]
t3=[[29, 29, 512]]
tff=tf.concat([t1,t2,t3] ,0)#가장 바깥 껍데기를 없애ㅗ 요소들을 합친다음 껍데기로 감싼 형태->3x3행렬이ㅣ
print(tff)
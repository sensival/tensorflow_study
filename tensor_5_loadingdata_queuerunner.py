# Queue Runner?
# 여러 파일을 리더로 읽고 Decoder로 해독(', 'seperate)-> 큐에 쌓고 배치만큼 조금씩 읽어옴. 메모리에 전체데이터 올리지 않음

import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

filename_queue = tf.train.string_input_producer(
    ["C:/Users/wogns/OneDrive/바탕 화면/tensorflow_study/data-02-test-score.txt.csv",
     "C:/Users/wogns/OneDrive/바탕 화면/tensorflow_study/data-03-test-score.txt.csv",
     "C:/Users/wogns/OneDrive/바탕 화면/tensorflow_study/data-01-test-score.txt.csv"],
     shuffle = False, name = 'filename_queue')# 파일리스트, 셔플여부, 큐이름

reader = tf.TextLineReader() # Binary등 다른 형식도 있음
key, value = reader.read(filename_queue) # key는 읽은 라인의 정보를 나타냅니다. 일반적으로 텍스트 파일의 각 라인에 대한 파일명 및 라인 번호 등을 나타냅니다.


record_defaults = [[0.],[0.],[0.],[0.]] #각각의 필드의 데이터형식 정해줄 수 있음. 없으면 생김
xy = tf.decode_csv(value, record_defaults = record_defaults) # value를 어떻게 파싱할지


train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size = 10)

'''
# shuffle_batch
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label], batch_size = batch_size, capacity = capacity, min_after_dequeue=min_after_dequeue) 
'''

X = tf.placeholder(tf.float32, shape=[None, 3]) # n개를 가질경우 [None,variable수]로 하면 원하는 만큼 가능
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost =tf.reduce_mean(tf.square(hypothesis -  Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                              feed_dict={X: x_batch, Y: y_batch}) # X에 행렬 변수x_data로
        
        if step % 100 ==0:
            print(step, "cost:", cost_val,"\nprediction:\n", hy_val)
            print(x_batch, y_batch)

coord.request_stop()
coord.join(threads)


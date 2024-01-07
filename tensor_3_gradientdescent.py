import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W를 랜덤값으로
W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))# 여기는 왜 sum 일까?--> 이유는 없는듯함.  평균이 일반적이긴하다

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y)* X) # cost fuction 미분식
descent = W - learning_rate * gradient
update = W.assign(descent)


sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})# update가 되는지
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))# step cost W 확인



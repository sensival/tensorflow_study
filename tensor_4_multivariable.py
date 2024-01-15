# 자주 사용하진 않음. Matrix가 더 선호
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()


x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 142.]
y_data = [152., 185., 180., 196., 142.]

# feed_dict로 던지기 위해 placeholder
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

cost =tf.reduce_mean(tf.square(hypothesis -  Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
        cost_val, hy_val, _, w1_val, w2_val, w3_val, b_val = sess.run([cost, hypothesis, train, w1, w2, w3, b],
                              feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
        if step % 100 ==0:
            print(step, "cost:", cost_val,"\nprediction:\n", hy_val)
            print("w1:", w1_val, "w2:", w2_val, "w3:", w3_val, "b:", b_val)

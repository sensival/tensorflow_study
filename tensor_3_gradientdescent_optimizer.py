import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

# 처음엔 이상한 W 설정
W = tf.Variable(5.0)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

'''
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y)* X) # cost fuction 미분식
descent = W - learning_rate * gradient
update = W.assign(descent)

'''
# 미분식 계산이 어려운 경우, 위의 코드 대신 optimizer 쓸수 있다. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(100):
    cost_val, W_val = sess.run([cost, W])  # cost와 W 값을 한 번에 계산
    print(step, cost_val, W_val)
    sess.run(train)

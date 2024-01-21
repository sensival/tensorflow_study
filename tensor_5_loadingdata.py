# data loading 하는 법
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

xy = np.loadtxt('C:/Users/wogns/OneDrive/바탕 화면/tensorflow_study/data-01-test-score.txt.csv', delimiter=',', dtype=np.float32, encoding='UTF-8')
x_data=xy[:,0:-1] # 모든 행의 마지막-1 인덱스까지
y_data=xy[:,[-1]] # 모든 행의 마지막 인덱스만 [:, -1:] 과 동일

'''
print(x_data)
print(y_data)
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 [ 89.  91.  90.]
 [ 96.  98. 100.]
 [ 73.  66.  70.]
 [ 53.  46.  55.]]
[[152.]
 [185.]
 [180.]
 [196.]
 [142.]
 [101.]]
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
for step in range(2001):
        cost_val, hy_val, _, w_val, b_val = sess.run([cost, hypothesis, train, W, b],
                              feed_dict={X: x_data, Y: y_data}) # X에 행렬 변수x_data로
        
        if step % 100 ==0:
            print(step, "cost:", cost_val,"\nprediction:\n", hy_val)
            print("W:", w_val, "b:", b_val)


# training 후 내 점수 예측
print("내 final score는", sess.run(hypothesis, feed_dict={X:[[100, 70, 101]]}))

print("친구들 final score는", sess.run(hypothesis, feed_dict={X:[[60, 70, 110],[90, 100, 80]]}))

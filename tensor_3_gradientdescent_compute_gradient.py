import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

# 처음엔 이상한 W 설정
W = tf.Variable(5.0)

hypothesis = X * W

# 미분식 수기로 계산
gradient = tf.reduce_mean((W * X - Y) * X) *2

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)


# gradient를 내가 수정하고 싶을때
gvs = optimizer.compute_gradients(cost)
# 수정한 뒤 다시 적용
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(100):
    # 우리가 계산한 미분식 결과(gradient)와  계산된 그래디언트(gvs)가 같은지 확인
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

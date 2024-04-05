import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2],
                   [2, 3],
                   [3, 1],
                   [4, 3],
                   [5, 3],
                   [6, 2]], dtype=np.float32)
y_data = np.array([[0],
                   [0],
                   [0],
                   [1],
                   [1],
                   [1]], dtype=np.float32)

# 입력 데이터를 받을 텐서 객체 생성
X = tf.constant(x_data)
Y = tf.constant(y_data)

# 가중치와 편향 변수 정의
W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 가설 함수 정의
def hypothesis(X):
    return tf.sigmoid(tf.matmul(X, W) + b)

# 손실 함수 정의
def cost_function(X, Y):
    predictions = hypothesis(X)
    loss = -tf.reduce_mean(Y * tf.math.log(predictions) + (1 - Y) * tf.math.log(1 - predictions))
    return loss

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 0.5 보다 크면 true 반환
def predicted(X):
    return tf.cast(hypothesis(X) > 0.5, dtype=tf.float32)

# 예측한 값이 얼마나 맞은가
def accuracy(X, Y):
    return tf.reduce_mean(tf.cast(tf.equal(predicted(X), Y), dtype=tf.float32))


# 학습 함수 정의
def train_step(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_function(X, Y)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


    # 학습 과정
for step in range(2000):
    train_step(X, Y)
    if step % 200 == 0:
        cost_val = cost_function(X, Y).numpy()
        print(step, cost_val)

h, c, a = hypothesis(x_data).numpy(), predicted(x_data).numpy(), accuracy(x_data, y_data).numpy()
print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
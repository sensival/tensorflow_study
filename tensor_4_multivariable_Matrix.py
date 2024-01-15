# Matrix 활용
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()



x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]] # [5, 3]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]] #[5, 1]
X = tf.placeholder(tf.float32, shape=[None, 3]) # n개를 가질경우 [None,variable수]로 하면 원하는 만큼 가능
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost =tf.reduce_mean(tf.square(hypothesis -  Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

cost_history = []
weight_history = [] # matplolib 그래프용


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
        cost_val, hy_val, _, w_val, b_val = sess.run([cost, hypothesis, train, W, b],
                              feed_dict={X: x_data, Y: y_data}) # X에 행렬 변수x_data로

        cost_history.append(cost_val)
        weight_history.append(w_val)
        
        if step % 100 ==0:
            print(step, "cost:", cost_val,"\nprediction:\n", hy_val)
            print("W:", w_val, "b:", b_val)


# Plot the cost over training steps
plt.plot(cost_history)
plt.xlabel('Training Steps')
plt.ylabel('Cost (Mean Squared Error)')
plt.title('Cost over Training Steps')
plt.show()

# Plot the weight values over training steps
weight_history = np.array(weight_history)
for i in range(3):  # Assuming 3 features in your data
    plt.plot(weight_history[:, i], label=f'W{i+1}')

plt.xlabel('Training Steps')
plt.ylabel('Weight Values')
plt.title('Weight Values over Training Steps')
plt.legend()
plt.show()

# Plot the final predicted values
final_prediction = sess.run(hypothesis, feed_dict={X: x_data})
plt.plot(y_data, label='Actual')
plt.plot(final_prediction, label='Predicted')
plt.xlabel('Data Points')
plt.ylabel('Target Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# Close the TensorFlow session
sess.close()

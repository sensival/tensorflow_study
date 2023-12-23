import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()# import tensorflow as tf 대신에
tf.set_random_seed(777)  # for reproducibility

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# placeholder로 입력
X= tf.placeholder(tf.float32)
Y= tf.placeholder(tf.float32)

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run( # '_'는 train 연산의 결과를 변수에 저장하지 않아도 된다는 뜻, 변수 지정하고 출력해도 none나옴
            [train, cost, W, b], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]}
        ) #train에 feed_dict의 값을 넘겨줌
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    
    # Plot the training data points
    plt.scatter([1, 2, 3], [1, 2, 3], color='red', marker='o', label='Training Data')

    # Plot the linear regression line using learned weights and bias
    x_values = [1, 2, 3]
    y_values = W_val * x_values + b_val
    plt.plot(x_values, y_values, color='blue', linewidth=2, label='Linear Regression')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression Model')

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()



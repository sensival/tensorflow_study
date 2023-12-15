'''
1. rank
rank 0 : scalar s=483
rank 1 : Vector v=[1,2,3]
rank 2 : matrix m=[[1,2,3],[4,5,6]]
rank 3 : 3- tensor t=[[[1],[2],[3]],[[4],[5],[6]]]
rank n : n-tensor

2. shape
[D0, D1, D2, ...Dn-1]

3.Types
tf.float32, tf.int32...
'''

import tensorflow as tf
# 0. 버전확인 2.6.2
print(tf.__version__)

'''tensorflow가 2.0버전으로 올라가면서, 더이상 Session모듈을 지원하지 않게되었습니다.
설치 오류가 아닙니다. 아래와같이 코드를 작성하시면 Session사용 가능합니다.
'''
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

# 1. Hello, TensorFlow! 출력
hello = tf1.constant("Hello, TensorFlow!")
sess1 = tf1.Session()
print(sess1.run(hello))
 
# 2-1.node1 + node2 = node3 그래프 빌드
node1 = tf1.constant(3.0, tf1.float32)
node2 = tf1.constant(4.0) # also tf.float32 implicitly
node3 = tf1.add(node1, node2) #node3 = node1+node2
print("node1:", node1, "node2:", node2)
print("node3: ", node3)

# 2-2.sess1.run
sess2 = tf1.Session()
print("sess2.run(node1, node2): ", sess2.run([node1, node2]))
print("sess2.run(node3): ", sess2.run(node3))

# 3-1.placeholder(값을 입력받을 때)
a = tf1.placeholder(tf1.float32)
b = tf1.placeholder(tf1.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess2.run(adder_node, feed_dict={a: 3, b: 4.5})) # a,b 값 입력 feed_dict
print(sess2.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))# n개의 값 입력
add_and_triple = adder_node * 3.
print(sess2.run(add_and_triple, feed_dict={a: 3, b:4.5}))


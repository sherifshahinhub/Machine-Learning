"""
# Creates a graph.
import tensorflow as tf
c = []
for d in ['/gpu:0']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(sum))
"""
"""
import tensorflow as tf
from datetime import datetime

shape = (5000, 5000)
device_name = "/GPU:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto()) as session:
        result = session.run(sum_operation)
        print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 2)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 2)
"""
import numpy as np
import time
from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a,b):
    return a+b

N = 32000000
A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)

start = time.time()
C = VectorAdd(A, B)
vector_add_time = time.time() - start
print('C[:5]={}'.format(str(C[:5])))
print('C[-5:]={}'.format(str(C[-5:])))
print('VectoreAdd took for {} seconds'.format(vector_add_time))
























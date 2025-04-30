import tensorflow as tf
import numpy as np

rank_0_tensor= tf.constant(1)

rank_1_tensor = tf.constant([1,2,3])

rank_2_tensor = tf.constant([[1,2,3], 
                             [1,1,1],
                             [2,2,2]],dtype=np.float16)

print(rank_2_tensor)                   

np.array(rank_2_tensor)


a=([[1,2,3],
    [1,1,1]])
b=([[1,1,1],
    [1,1,1]])

print("Add:", tf.add(a,b))
print("Multiply: ", tf.multiply(a,b))
print("Matrix:", tf.matmul(a,b))


four=tf.zeros([3,4,5,6])

print(four.ndim)
print(four.shape[0])
print(four.shape)
print(four.size)

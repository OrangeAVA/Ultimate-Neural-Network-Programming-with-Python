import tensorflow as tf


a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`


print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")


## The above operation can be written in other way as well


print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication


#Output: 
# [[2 3]
#  [4 5]], shape=(2, 2), dtype=int32) 


# tf.Tensor(
# [[1 2]
#  [3 4]], shape=(2, 2), dtype=int32) 


# tf.Tensor(
# [[3 3]
#  [7 7]], shape=(2, 2), dtype=int32) 

#Few other operations:
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])


# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.math.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))


#Output: 
# tf.Tensor(10.0, shape=(), dtype=float32)
# tf.Tensor([1 0], shape=(2,), dtype=int64)
# tf.Tensor(
# [[2.6894143e-01 7.3105860e-01]
#  [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)

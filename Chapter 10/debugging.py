import tensorflow as tf
import pdb


def divide_tensors(a, b):
    result = tf.divide(a, b)
    return result


# Define input tensors
a = tf.constant([1, 2, 3], dtype=tf.float32)
b = tf.constant([0, 2, 4], dtype=tf.float32)


# Call the function
output = divide_tensors(a, b)


# Create a TensorFlow session
with tf.Session() as sess:
    try:
        # Start the debugger
        pdb.set_trace()


        # Run the session and print the output
        result = sess.run(output)
        print("Result:", result)
    except tf.errors.InvalidArgumentError as e:
        print("Error:", e)
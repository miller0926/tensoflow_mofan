#placeholder 传入值
import tensorflow as tf

input1 = tf.placeholder(tf.float32)#可控制格式，如tf.placeholder(tf.float32,[2,2])
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))
    #print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
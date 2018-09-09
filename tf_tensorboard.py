import tensorflow as tf
import numpy as np
import matplotlib as plt

def add_layer(inputs, in_size, out_size, activation_function = None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W')#随机产生矩阵大小
            tf.histogram_summary(layer_name+'/Weights',Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)#biases 1维列表
            tf.histogram_summary(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b,)
        tf.histogram_summary(layer_name+'/outputs',outputs)
        return outputs


#生成x_data,noise,y_data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('input'):
    #传入xs,ys
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'x_input')

#添加神经层
l1 = add_layer(xs,1,10,n_layer = 1, activation_function = tf.nn.relu)
#预测结果
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    #定义loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

    tf.scalar_summary('loss',loss)

with tf.name_scope('train'):
    #训练神经网络
    train_step = tf.train.GrandientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

merged = tf.merge_all_summaries()
writer =tf.train.SummaryWriter("F:\logs",sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)#绘制断点形式的x_data,y_data
plt.ion()#连续绘图
plt.show()

#训练1000步，每50步输出一次loss
for i in range(1000):
    sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
    if i%50 == 0:
        # #print(sess.run(loss,feed_dict ={xs:x_data,ys:y_data}))
        # #忽略掉第一次没有绘制线的错误
        # try：
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value = sess.run(prediction.feed_dict = {xs:x_data})
        # lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
        # plt.pause(0.1)
        result =sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
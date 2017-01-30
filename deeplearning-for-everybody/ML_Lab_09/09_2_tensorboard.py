import tensorflow as tf
import numpy as np

xy = np.loadtxt("xor_train.txt", unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1],(4,1))

X = tf.placeholder(tf.float32, name="X-input")
Y = tf.placeholder(tf.float32, name="Y-input")

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name="weight1")
W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0), name="weight2")
W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name="weight3")

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

w1_hist = tf.summary.histogram("weights1", W1)
w2_hist = tf.summary.histogram("weights2", W2)
w3_hist = tf.summary.histogram("weights3", W3)

b1_hist = tf.summary.histogram("Biases1", b1)
b2_hist = tf.summary.histogram("Biases2", b2)
b3_hist = tf.summary.histogram("Biases3", b3)

y_hist = tf.summary.histogram("y", Y)

with tf.name_scope("layer2") as scpoe :
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
    
with tf.name_scope("layer3") as scpoe :    
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
    
with tf.name_scope("layer4") as scpoe :
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

with tf.name_scope("cost") as scpoe :
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

a = tf.Variable(0.1)

with tf.name_scope("train") as scpoe :
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess :

    #tensorboard merge
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

    sess.run(init)
    
    for step in range(20001) :
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        
        if step % 2000 == 0 :
            summary, _ = sess.run([merged, train], feed_dict={X:x_data, Y:y_data})
            writer.add_summary(summary, step)
    
    #hypothesis = 0~1사이의값 -> 0.5를 더하여 floor하면 0 또는 1로 나타남
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    
    #평균구하기
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print("Accruacy : ", accuracy.eval({X:x_data, Y:y_data}))
    
    
#in cmd
#tensorboard --logdir=./logs
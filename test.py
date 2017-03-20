import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
# print node1, node2

sess = tf.Session()
# print sess.run([node1,node2])

node3 = tf.add(node1,node2)
# print "node3: ",node3
# print "sess.run(node3): ",sess.run(node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = tf.add(a,b)

# print sess.run(c,{a:[1,2],b:[1,2]})
# print sess.run(c,{a:1,b:2})

W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x+b

init = tf.global_variables_initializer()
sess.run(init)

# print sess.run(linear_model,{x:[1,2,3,4]})

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
# print sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]})

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

# print sess.run([W,b])

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
# print estimator.evaluate(input_fn=input_fn)

def model(features,labels,mode):
    W = tf.get_variable("W",[1],dtype = tf.float64)
    b = tf.get_variable("b",[1],dtype = tf.float64)
    y = W*features['x']+b

    loss = tf.reduce_sum(tf.square(y-labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),tf.assign_add(global_step,1))

    return tf.contrib.learn.ModelFnOps(mode=mode, predictions = y, loss = loss, train_op = train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x},y,4,num_epochs=1000)

estimator.fit(input_fn=input_fn,steps=1000)
print estimator.evaluate(input_fn=input_fn,steps=10)




























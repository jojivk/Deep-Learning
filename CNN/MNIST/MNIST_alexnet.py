###############################################################
#    A Pseudo AlexNet. Since MNIST is smaller in size 28x28
#    We use the 2-8 layers of Alexnet as the network.
# Jojimon Varghese
###############################################################


from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

###############################################################
# 1. Read the data set
###############################################################
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

###############################################################
# 3. Define hyper parameters and some constants.
#    Helps to experiment by changing them
###############################################################
num_steps=2000
lr = 0.005
classes = 10
max_steps = 2000
batch_size = 128
dout=0.25

###############################################################
# 
###############################################################
def local_response_normalization(lay) :
  lay1 = tf.nn.local_response_normalization(lay)
  out = tf.nn.elu(lay1)
  return out;

###############################################################
# 4. Define the network using tf.layers
#    A Pseudo AlexNet. Since MNIST is smaller in size 28x28
#    We use the 2-8 layers of Alexnet as the network.
#    4 Convloutional layers. 1 normalization layer
#    3 Fully connected layers
# Since MNIST is gray scale  and possible features are less,
# I have reduce the no of features in
# each layer by a factor of 4 
#   256-64, 384-96...
###############################################################
def network(X, dropout=0.5, is_training=True) :
  lay0 = tf.reshape(X, [-1, 28, 28, 1])

  lay1 = tf.layers.conv2d(lay0, 64, 5, activation=None)
  lay1 = tf.layers.max_pooling2d(lay1, 3, 2)
  lay2 = local_response_normalization(lay1);

  lay3 = tf.layers.conv2d(lay2, 96, 3, padding="same", activation=tf.nn.elu)
  lay4 = tf.layers.conv2d(lay3, 96, 3, padding="same", activation=tf.nn.elu)
  lay5 = tf.layers.conv2d(lay4, 64, 3, padding="same", activation=tf.nn.elu)
  lay5 = tf.layers.max_pooling2d(lay5, 3, 2)

  fc6 = tf.contrib.layers.flatten(lay5)
  fc7 = tf.layers.dense(fc6, 1024)
  fc8 = tf.layers.dense(fc7, 1024)
  out = tf.layers.dense(fc8, classes)

  return out;

###############################################################
# 5. Build the model fun to be passed to the estimator.
#    This implments the logits. loss_fn and using them
#    creates the estimator spec, that is returned.
###############################################################
def model_fn(features, labels, mode) :
  
  X= features['images']
  logits = network(X, dout, mode==tf.estimator.ModeKeys.TRAIN)

  pred = tf.argmax(logits, axis=1)
  prob = tf.nn.softmax(logits)

  if mode == tf.estimator.ModeKeys.PREDICT :
    return tf.estimator.EstimatorSpec(mode, predictions=pred)

  loss_fn = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
  trainer = tf.train.AdagradOptimizer(lr).minimize(loss_fn,
                    global_step=tf.train.get_global_step()) 

  acc = tf.metrics.accuracy(labels=labels, predictions=pred)

  eSpec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred,
                                     loss=loss_fn, train_op=trainer,
                                     eval_metric_ops={'accuracy': acc})

  return eSpec;

###############################################################
# 6. Define datsets for train evaluation and testing
###############################################################
train_ip = tf.estimator.inputs.numpy_input_fn(
                     x={'images':mnist.train.images}, y=mnist.train.labels,
                     batch_size=batch_size, num_epochs=None, shuffle=True)
eval_ip = tf.estimator.inputs.numpy_input_fn(
                     x={'images':mnist.test.images}, y=mnist.test.labels, shuffle=False)
test_images = mnist.test.images[0:4]
test_ip = tf.estimator.inputs.numpy_input_fn(
                     x={'images':test_images}, shuffle=False)

# for printing log info
tf.logging.set_verbosity(tf.logging.INFO)
###############################################################
# 7. Use estimator spec to build the model
#     Then train the model using training data and evalute it
#     Finally test with test data
###############################################################
model = tf.estimator.Estimator(model_fn)
model.train(train_ip, steps=max_steps)
print("\n\n---------------------------All training done---------------------")
print("---------------------------Eval with test set--------------------\n\n")

model.evaluate(eval_ip)
op= list(model.predict(test_ip))

###############################################################
# 8. Display a few sample predictions
###############################################################
for i in range(4):
  plt.imshow(np.reshape(test_images[i], [28, 28]))
  plt.show()
  print("Model prediction", op[i])

print(op)
 

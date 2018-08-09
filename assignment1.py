from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
from six.moves import range

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '/Users/rakadalal/Desktop/Deep_learning/' # Change me to store data elsewhere

num_classes = 10
np.random.seed(133)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation:', valid_dataset.shape, valid_labels.shape)
# print('Testing:', test_dataset.shape, test_labels.shape)

# image_size = 28
# num_labels = 10

# def reformat(dataset, labels):
#   dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
#   # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
#   labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
#   return dataset, labels
# train_dataset, train_labels = reformat(train_dataset, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
# test_dataset, test_labels = reformat(test_dataset, test_labels)
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
# train_subset = 10000

# graph = tf.Graph()
# with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  # tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  # tf_train_labels = tf.constant(train_labels[:train_subset])
  # tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)


  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  # weights = tf.Variable(
  #   tf.truncated_normal([image_size * image_size, num_labels]))
  # biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  # logits = tf.matmul(tf_train_dataset, weights) + biases
  # loss = tf.reduce_mean(
  #   tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
#   train_prediction = tf.nn.softmax(logits)
#   valid_prediction = tf.nn.softmax(
#     tf.matmul(tf_valid_dataset, weights) + biases)
#   test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

# num_steps = 801

# def accuracy(predictions, labels):
#   return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
#           / predictions.shape[0])

# with tf.Session(graph=graph) as session:
#   # This is a one-time operation which ensures the parameters get initialized as
#   # we described in the graph: random weights for the matrix, zeros for the
#   # biases. 
#   tf.global_variables_initializer().run()
#   print('Initialized')
#   for step in range(num_steps):
#     # Run the computations. We tell .run() that we want to run the optimizer,
#     # and get the loss value and the training predictions returned as numpy
#     # arrays.
#     _, l, predictions = session.run([optimizer, loss, train_prediction])
#     if (step % 100 == 0):
#       print('Loss at step %d: %f' % (step, l))
#       print('Training accuracy: %.1f%%' % accuracy(
#         predictions, train_labels[:train_subset, :]))
#       # Calling .eval() on valid_prediction is basically like calling run(), but
#       # just to get that one numpy array. Note that it recomputes all its graph
#       # dependencies.
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

# batch_size = 128

# graph = tf.Graph()
# with graph.as_default():

#   # Input data. For the training data, we use a placeholder that will be fed
#   # at run time with a training minibatch.
#   tf_train_dataset = tf.placeholder(tf.float32,
#                                     shape=(batch_size, image_size * image_size))
#   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#   tf_valid_dataset = tf.constant(valid_dataset)
#   tf_test_dataset = tf.constant(test_dataset)
  
#   # Variables.
#   weights = tf.Variable(
#     tf.truncated_normal([image_size * image_size, 1024]))
#   biases = tf.Variable(tf.zeros([1024]))

#   # Variables.
#   weights2 = tf.Variable(
#     tf.truncated_normal([1024, num_labels]))
#   biases2 = tf.Variable(tf.zeros([num_labels]))
  
#   # Training computation.
#   logits = tf.matmul(tf_train_dataset, weights) + biases
#   hidden1=tf.nn.relu(logits, name=None)


#   logits = tf.matmul(hidden1, weights2) + biases2
#   loss = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
#   # Optimizer.
#   optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
#   # Predictions for the training, validation, and test data.
#   train_prediction = tf.nn.softmax(logits)
#   valid_prediction = tf.nn.softmax(
#     tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases), weights2) + biases2)
#   test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases), weights2) + biases2)


# num_steps = 3001

# with tf.Session(graph=graph) as session:
#   tf.global_variables_initializer().run()
#   print("Initialized")
#   for step in range(num_steps):
#     # Pick an offset within the training data, which has been randomized.
#     # Note: we could use better randomization across epochs.
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     # Generate a minibatch.
#     batch_data = train_dataset[offset:(offset + batch_size), :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     # Prepare a dictionary telling the session where to feed the minibatch.
#     # The key of the dictionary is the placeholder node of the graph to be fed,
#     # and the value is the numpy array to feed to it.
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 500 == 0):
#       print("Minibatch loss at step %d: %f" % (step, l))
#       print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#       print("Validation accuracy: %.1f%%" % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# train_dataset = train_dataset[0:5000,:,:] 
# train_labels=train_labels[0:5000]
# print('Training:', train_dataset.shape, train_labels.shape)

# nsamples, nx, ny = train_dataset.shape
# d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

# nsamples, nx, ny = test_dataset.shape
# d2_test_dataset = test_dataset.reshape((nsamples,nx*ny))

# logreg = LogisticRegression()
# logreg.fit(d2_train_dataset, train_labels)

# y_pred = logreg.predict(d2_test_dataset)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(d2_test_dataset, test_labels)))

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])



# beta_val = np.logspace(-4, -2, 20)
# batch_size = 128
# accuracy_val = []

# logistic model

# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32,
#                                       shape=(batch_size, image_size * image_size))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#     beta_regul = tf.placeholder(tf.float32)

#     # Variables.
#     weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))

#     # Training computation.
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + beta_regul * tf.nn.l2_loss(weights)

#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

# num_steps = 3001

# for beta in beta_val:
#     with tf.Session(graph=graph) as session:
#         tf.initialize_all_variables().run()
#         for step in range(num_steps):
#             # Pick an offset within the training data, which has been randomized.
#             # Note: we could use better randomization across epochs.
#             offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#             # Generate a minibatch.
#             batch_data = train_dataset[offset:(offset + batch_size), :]
#             batch_labels = train_labels[offset:(offset + batch_size), :]
#             # Prepare a dictionary telling the session where to feed the minibatch.
#             # The key of the dictionary is the placeholder node of the graph to be fed,
#             # and the value is the numpy array to feed to it.
#             feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: beta}
#             _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#             # if (step % 500 == 0):
#             #     print("Minibatch loss at step %d: %f" % (step, l))
#             #     print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             #     print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
#         print("L2 regularization(beta=%.5f) Test accuracy: %.1f%%" % (
#             beta, accuracy(test_prediction.eval(), test_labels)))

#         accuracy_val.append(accuracy(test_prediction.eval(), test_labels))

# print('Best beta=%f, accuracy=%.1f%%' % (beta_val[np.argmax(accuracy_val)], max(accuracy_val)))
# plt.semilogx(beta_val, accuracy_val)
# plt.grid(True)
# plt.title('Test accuracy by regularization (logistic)')
# plt.show()

# NN model
# batch_size = 128
# hidden_size = 1024

# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#     tf_beta = tf.placeholder(tf.float32)

#     # Variables.
#     W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size]))
#     b1 = tf.Variable(tf.zeros([hidden_size]))

#     W2 = tf.Variable(tf.truncated_normal([hidden_size, num_labels]))
#     b2 = tf.Variable(tf.zeros([num_labels]))

#     # Training computation.
#     y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
#     logits = tf.matmul(y1, W2) + b2

#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

#     loss = loss + tf_beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))

#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)

#     y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
#     valid_logits = tf.matmul(y1_valid, W2) + b2
#     valid_prediction = tf.nn.softmax(valid_logits)




#     y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
#     test_logits = tf.matmul(y1_test, W2) + b2
#     test_prediction = tf.nn.softmax(test_logits)


# for beta in beta_val:
# 	with tf.Session(graph=graph) as session:
# 	  tf.global_variables_initializer().run()
# 	  print("Initialized")
# 	  for step in range(num_steps):
# 	    # Pick an offset within the training data, which has been randomized.
# 	    # Note: we could use better randomization across epochs.
# 	    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
# 	    # Generate a minibatch.
# 	    batch_data = train_dataset[offset:(offset + batch_size), :]
# 	    batch_labels = train_labels[offset:(offset + batch_size), :]
# 	    # Prepare a dictionary telling the session where to feed the minibatch.
# 	    # The key of the dictionary is the placeholder node of the graph to be fed,
# 	    # and the value is the numpy array to feed to it.
# 	    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_beta: beta}
# 	    _, l, predictions = session.run(
# 	      [optimizer, loss, train_prediction], feed_dict=feed_dict)
# 	    print("L2 regularization(beta=%.5f) Test accuracy: %.1f%%" % (
#             beta, accuracy(test_prediction.eval(), test_labels)))

#         accuracy_val.append(accuracy(test_prediction.eval(), test_labels))

# print('Best beta=%f, accuracy=%.1f%%' % (beta_val[np.argmax(accuracy_val)], max(accuracy_val)))
# plt.semilogx(beta_val, accuracy_val)
# plt.grid(True)
# plt.title('Test accuracy by regularization (logistic)')
# plt.show()


# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting.
# Restrict your training data to just a few batches. What happens?
#
# ---

# few_batch_size = batch_size * 5
# small_train_dataset = train_dataset[:few_batch_size, :]
# small_train_labels = train_labels[:few_batch_size, :]

# print('Training set', small_train_dataset.shape, small_train_labels.shape)

# num_steps = 3001

# with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     print("Initialized")
#     for step in range(num_steps):
#         # Pick an offset within the training data, which has been randomized.
#         # Note: we could use better randomization across epochs.
#         offset = (step * batch_size) % (small_train_labels.shape[0] - batch_size)
#         # Generate a minibatch.
#         batch_data = small_train_dataset[offset:(offset + batch_size), :]
#         batch_labels = small_train_labels[offset:(offset + batch_size), :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it.
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta: 0.001438}
#         _, l, predictions = session.run(
#             [optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 500 == 0):
#             print("Minibatch loss at step %d: %f" % (step, l))
#             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             print("Validation accuracy: %.1f%%" % accuracy(
#                 valid_prediction.eval(), valid_labels))
#     print("Overfitting with small dataset Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network.
# Remember: Dropout should only be introduced during training, not evaluation,
# otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that,
# but you have to make sure it's only inserted during training.
#
# What happens to our extreme overfitting case?
#
# ---


# batch_size = 128
# hidden_size = 1024

# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#     tf_beta = tf.placeholder(tf.float32)

#     # Variables.
#     W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size]))
#     b1 = tf.Variable(tf.zeros([hidden_size]))

#     W2 = tf.Variable(tf.truncated_normal([hidden_size, num_labels]))
#     b2 = tf.Variable(tf.zeros([num_labels]))

#     # Training computation.
#     y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
#     y1 = tf.nn.dropout(y1, 0.5)  # Dropout
#     logits = tf.matmul(y1, W2) + b2

#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

#     loss = loss + tf_beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))

#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)

#     y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
#     valid_logits = tf.matmul(y1_valid, W2) + b2
#     valid_prediction = tf.nn.softmax(valid_logits)

#     y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
#     test_logits = tf.matmul(y1_test, W2) + b2
#     test_prediction = tf.nn.softmax(test_logits)

# # Let's run it:
# num_steps = 3001

# with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     print("Initialized")
#     for step in range(num_steps):
#         # Pick an offset within the training data, which has been randomized.
#         # Note: we could use better randomization across epochs.
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         # Generate a minibatch.
#         batch_data = train_dataset[offset:(offset + batch_size), :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it.
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta: 0.001438}
#         _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 500 == 0):
#             print("Minibatch loss at step %d: %f" % (step, l))
#             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             print("Validation accuracy: %.1f%%" % accuracy(
#                 valid_prediction.eval(), valid_labels))
#     print("Dropout Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# few_batch_size = batch_size * 5
# small_train_dataset = train_dataset[:few_batch_size, :]
# small_train_labels = train_labels[:few_batch_size, :]

# print('Training set', small_train_dataset.shape, small_train_labels.shape)

# num_steps = 3001

# with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     print("Initialized")
#     for step in range(num_steps):
#         # Pick an offset within the training data, which has been randomized.
#         # Note: we could use better randomization across epochs.
#         offset = (step * batch_size) % (small_train_labels.shape[0] - batch_size)
#         # Generate a minibatch.
#         batch_data = small_train_dataset[offset:(offset + batch_size), :]
#         batch_labels = small_train_labels[offset:(offset + batch_size), :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it.
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta: 0.001438}
#         _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 500 == 0):
#             print("Minibatch loss at step %d: %f" % (step, l))
#             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             print("Validation accuracy: %.1f%%" % accuracy(
#                 valid_prediction.eval(), valid_labels))
#     print("Dropout with small dataset Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# ---
# Problem 4
# ---------
#
# Try to get the best performance you can using a multi-layer model!
# The best reported test accuracy using a deep network is [97.1%]
# (http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
#
# One avenue you can explore is to add multiple layers.
#
# Another one is to use learning rate decay:
#
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#  ---
#

# batch_size = 128
# fc1_size = 4096
# fc2_size = 2048
# fc3_size = 128

# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32,
#                                       shape=(batch_size, image_size * image_size))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#     tf_beta = tf.placeholder(tf.float32)
#     global_step = tf.Variable(0)  # count the number of steps taken.

#     # Variables.
#     # stddev is very important!!!
#     W1 = tf.Variable(
#         tf.truncated_normal([image_size * image_size, fc1_size], stddev=np.sqrt(2.0 / (image_size * image_size))))
#     b1 = tf.Variable(tf.zeros([fc1_size]))

#     W2 = tf.Variable(tf.truncated_normal([fc1_size, fc2_size], stddev=np.sqrt(2.0 / fc1_size)))
#     b2 = tf.Variable(tf.zeros([fc2_size]))

#     W3 = tf.Variable(tf.truncated_normal([fc2_size, fc3_size], stddev=np.sqrt(2.0 / fc2_size)))
#     b3 = tf.Variable(tf.zeros([fc3_size]))

#     W4 = tf.Variable(tf.truncated_normal([fc3_size, num_labels], stddev=np.sqrt(2.0 / fc3_size)))
#     b4 = tf.Variable(tf.zeros([num_labels]))

#     # Training computation.
#     y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
#     # y1 = tf.nn.dropout(y1, 0.5)

#     y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
#     # y2 = tf.nn.dropout(y2, 0.5)

#     y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
#     # y3 = tf.nn.dropout(y3, 0.5)

#     logits = tf.matmul(y3, W4) + b4

#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

#     loss = loss + tf_beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
#                              tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

#     # Optimizer
#     learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.7, staircase=True)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)

#     y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
#     y2_valid = tf.nn.relu(tf.matmul(y1_valid, W2) + b2)
#     y3_valid = tf.nn.relu(tf.matmul(y2_valid, W3) + b3)
#     valid_logits = tf.matmul(y3_valid, W4) + b4
#     valid_prediction = tf.nn.softmax(valid_logits)

#     y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
#     y2_test = tf.nn.relu(tf.matmul(y1_test, W2) + b2)
#     y3_test = tf.nn.relu(tf.matmul(y2_test, W3) + b3)
#     test_logits = tf.matmul(y3_test, W4) + b4
#     test_prediction = tf.nn.softmax(test_logits)

# # Let's run it:
# num_steps = 12001

# with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     print("Initialized")
#     for step in range(num_steps):
#         # Pick an offset within the training data, which has been randomized.
#         # Note: we could use better randomization across epochs.
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         # Generate a minibatch.
#         batch_data = train_dataset[offset:(offset + batch_size), :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it.
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta: 0.001438}
#         _, l, predictions = session.run(
#             [optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 500 == 0):
#             print("Minibatch loss at step %d: %f" % (step, l))
#             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             print("Validation accuracy: %.1f%%" % accuracy(
#                 valid_prediction.eval(), valid_labels))
#     print("Final Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


##Assignment 4

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# batch_size = 16
# patch_size = 5
# depth = 16
# num_hidden = 64

# graph = tf.Graph()

# with graph.as_default():

#   # Input data.
#   tf_train_dataset = tf.placeholder(
#     tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#   tf_valid_dataset = tf.constant(valid_dataset)
#   tf_test_dataset = tf.constant(test_dataset)
  
#   # Variables.
#   layer1_weights = tf.Variable(tf.truncated_normal(
#       [patch_size, patch_size, num_channels, depth], stddev=0.1))
#   layer1_biases = tf.Variable(tf.zeros([depth]))
#   layer2_weights = tf.Variable(tf.truncated_normal(
#       [patch_size, patch_size, depth, depth], stddev=0.1))
#   layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
#   layer3_weights = tf.Variable(tf.truncated_normal(
#       [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#   layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
#   layer4_weights = tf.Variable(tf.truncated_normal(
#       [num_hidden, num_labels], stddev=0.1))
#   layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
#   def model(data):
#     conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer1_biases)
#     conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer2_biases)
#     shape = hidden.get_shape().as_list()
#     reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#     hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#     return tf.matmul(hidden, layer4_weights) + layer4_biases
  
#   # Training computation.
#   logits = model(tf_train_dataset)
#   loss = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
#   # Optimizer.
#   optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
#   # Predictions for the training, validation, and test data.
#   train_prediction = tf.nn.softmax(logits)
#   valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#   test_prediction = tf.nn.softmax(model(tf_test_dataset))

# num_steps = 1001

# with tf.Session(graph=graph) as session:
#   tf.global_variables_initializer().run()
#   print('Initialized')
#   for step in range(num_steps):
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 50 == 0):
#       print('Minibatch loss at step %d: %f' % (step, l))
#       print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


##Problem 1

# batch_size = 16
# patch_size = 5
# depth = 16
# num_hidden = 64

# graph = tf.Graph()

# with graph.as_default():
#     # Input data.
#     tf_train_dataset = tf.placeholder(
#         tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)

#     # Variables.
#     layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
#     layer1_biases = tf.Variable(tf.zeros([depth]))

#     layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
#     layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

#     layer3_weights = tf.Variable(
#         tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#     layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

#     layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
#     layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


#     # Model.
#     def model(data):
#         conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases)
#         pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#         conv2 = tf.nn.relu(tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases)
#         pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#         shape = pool2.get_shape().as_list()
#         reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
#         fc1 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#         return tf.matmul(fc1, layer4_weights) + layer4_biases


#     # Training computation.
#     logits = model(tf_train_dataset)
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#     test_prediction = tf.nn.softmax(model(tf_test_dataset))

# num_steps = 1001
# with tf.Session(graph=graph) as session:
#   tf.global_variables_initializer().run()
#   print('Initialized')
#   for step in range(num_steps):
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 50 == 0):
#       print('Minibatch loss at step %d: %f' % (step, l))
#       print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem 2
# ---------
# 
# Try to get the best performance you can using a convolutional net.
# Look for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, 
# adding Dropout, and/or adding learning rate decay.
# Deep MNIST for expert
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
# ---

batch_size = 16
patch_size = 5
depth = 32
num_hidden = 64
num_steps = 30001

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(
        tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data, keep_prob):
        conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        fc1 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        y_conv = tf.matmul(fc1_drop, layer4_weights) + layer4_biases

        return y_conv


    # Training computation.
    logits = model(tf_train_dataset, 0.5)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    loss = loss + 0.001 * (tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases))

    # Optimizer.
    learning_rate = tf.train.exponential_decay(1e-1, global_step, num_steps, 0.7, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 300 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Lenet 5 Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


import numpy as np
import keras
import md5
import os

class BottleneckDataGenerator(object):
  'Generates bottleneck features for Keras'
  def __init__(self, directory, bottlenecks, batch_size=32, shuffle=True, seed=None, val_split=0):
    self.train_generator = None
    self.validation_generator = None
    self.class_indices = {}
    self.labels = {}
    self.files = {}
    self.partition = {'train': [], 'validation': []}

    # get the list of paths from the source directory
    # ensure category subdirectories are sorted
    for catindex, category in enumerate(sorted(os.listdir(directory))):
      self.class_indices[catindex] = category
      catdir = next(os.walk(directory+category))[2]
      train_count = int(round((1-val_split)*len(catdir)))
      if shuffle == True:
        np.random.seed(seed)
        np.random.shuffle(catdir)
      for fileindex, file in enumerate(catdir):
        if not file.lower().endswith(('.png', '.jpg', '.bmp', '.ppm', '.tif')):
          continue
        id = md5.new(category+"-"+file).hexdigest()
        self.labels[id] = catindex
        self.files[id] = file
        # partition into train and validation sets if val_split is not 0
        if fileindex < train_count:
          self.partition['train'].append(id)
        else:
          self.partition['validation'].append(id)

    self.train_generator = BottleneckIterator(self.partition['train'], bottlenecks, self.class_indices, self.labels, self.files, batch_size=batch_size, shuffle=shuffle, seed=seed)
    if val_split:
      self.validation_generator = BottleneckIterator(self.partition['validation'], bottlenecks, self.class_indices, self.labels, self.files, batch_size=batch_size, shuffle=shuffle, seed=seed)

  def flow_from_directory(self, subset=None):
    if subset == None or subset == "training":
      return self.train_generator
    else:
      return self.validation_generator 

class BottleneckIterator(keras.utils.Sequence):
  'Iterates through a list of files and returns appropriate bottleneck files as samples organized in batches'
  def __init__(self, indices, bottlenecks, class_indices, labels, files, batch_size=32, shuffle=True, seed=None):
    'Initialization'
    self.indices = indices
    self.bottlenecks = bottlenecks
    self.class_indices = class_indices
    self.labels = labels
    self.files = files
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.input_shape = None
    self.samples = len(self.indices)

    self.on_epoch_end()
    self.determine_input_shape()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.indices) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indices of the batch
    indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

    # Generate data
    X, y = self.__data_generation(indices)

    return X, y

  def determine_input_shape(self):
    'Loads first bottleneck and determines the shape'
    b = np.load(self.bottlenecks + "/" + self.class_indices[self.labels[self.indices[0]]] + "/" + self.files[self.indices[0]] + '.npy')
    self.input_shape = b.shape

  def on_epoch_end(self):
    'Updates indices after each epoch'
    if self.shuffle == True:
      np.random.seed(self.seed)
      np.random.shuffle(self.indices)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    if self.batch_size > len(list_IDs_temp):
      tensor_size = len(list_IDs_temp)
    else:
      tensor_size = self.batch_size
    X = np.empty((tensor_size, ) + self.input_shape[1:])
    y = np.empty((tensor_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      # Store sample
      X[i,] = np.load(self.bottlenecks + "/" + self.class_indices[self.labels[ID]] + "/" + self.files[ID] + '.npy')

      # Store class
      y[i] = self.labels[ID]

    return X, keras.utils.to_categorical(y, num_classes=len(self.class_indices))

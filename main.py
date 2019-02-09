from __future__ import print_function

# Networks
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator

# Layers
from keras.layers import *
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from generators import BottleneckDataGenerator
from vis.visualization import visualize_activation,visualize_saliency,overlay,visualize_cam
from vis.utils import utils as visutils

#Tensorboard
from tensorboard import TrainValTensorBoard

#Tensorflow
import tensorflow as tf

# Utils
import numpy as np
import random as rn
import argparse
import os
import cv2
import time
import math
import progressbar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Files
import utils

# For boolean input from the command line
def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \
                                                               Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--image_dir', type=str, default=None, help='Folder of images you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, required=True, help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=float, default=0.0, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')
parser.add_argument('--model', type=str, default="MobileNet", help='Your pre-trained classification model of choice')
parser.add_argument('--val_split', type=float, default=0.1, help='Percentage of dataset to use for validation')
parser.add_argument('--bottlenecks', type=str2bool, default=False, help='Wether to use bottlenecks or not (mutually exclusive with data augmentation options)')
parser.add_argument('--only_luma', type=str2bool, default=True, help='Convert images to luminance (0.21, 0.72, 0.07)')
parser.add_argument('--skip_bottleneck_check', type=str2bool, default=False, help='Set to true to skip generating/checking for bottlenecks (if sure they\'re all accounted for)')
args = parser.parse_args()


# Global settings
BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
TRAIN_DIR = args.dataset
DATASET_NAME = os.path.normpath(args.dataset).replace('..', 'parent').replace('/', '-')
USE_BOTTLENECKS = args.bottlenecks
INIT_LR = 1e-3
WEIGHTS_PATH = "./checkpoints/" + args.model + "_model_weights.h5"
MODEL_PATH = "./checkpoints/" + args.model + "_model.json"
FINAL_WEIGHTS_PATH = "./models/" + args.model + "_model_weights.h5"
FINAL_MODEL_PATH = "./models/" + args.model + "_model.json"

os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(1337)

# Setting the seed for python random numbers
rn.seed(1337)

# Setting the graph-level random seed.
tf.set_random_seed(1337)

preprocessing_function = None
base_model = None

def recall(y_true, y_pred):
  #Recall metric.

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)),axis=0)
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision(y_true, y_pred):
  #Precision metric.

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)),axis=0)
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def accuracy(y_true, y_pred):
  correct_prediction = K.equal(K.round(result_tensor), ground_truth_tensor)
  #accuracy = 
  pass

def save_bottleneck_features():
  model = base_model
  generator, _ = prepare_data_generators(val_split=0, batch_size=1, class_mode=None, shuffle=False)

  print("Bottlenecking", str(len(generator.filenames)) + " features.")

  total = 0
  with progressbar.ProgressBar(max_value=len(generator.filenames)) as bar:
    batch_index = 0
    for batch in generator:
      filename = get_bottleneck_base_path()+'/'+generator.filenames[batch_index]+'.npy'

      if not os.path.isfile(filename):
        bottleneck = model.predict(batch, verbose=0)
        if not os.path.exists(os.path.dirname(filename)):
          os.makedirs(os.path.dirname(filename))
        np.save(filename, bottleneck)

      if batch_index > (generator.batch_index):
        break
      batch_index = generator.batch_index        

      total += 1
      bar.update(total)

def prepare_bottleneck_data_generators(val_split, batch_size):
  bottleneck_datagen = BottleneckDataGenerator(
    TRAIN_DIR, 
    get_bottleneck_base_path(), 
    batch_size=batch_size, 
    shuffle=True, 
    val_split=val_split
  )

  train_generator = bottleneck_datagen.flow_from_directory(subset='training')
  
  if val_split:
    validation_generator = bottleneck_datagen.flow_from_directory(subset='validation')
  else:
    validation_generator = None

  return train_generator, validation_generator

def get_bottleneck_base_path():
  return 'bottlenecks/'+args.model+'-'+str(WIDTH)+'-'+str(HEIGHT)+'/'+DATASET_NAME

def grayscale_preprocessing_shim(x, **kwargs):
  x = (0.21 * x[:,:,:1]) + (0.72 * x[:,:,1:2]) + (0.07 * x[:,:,-1:])
  return preprocessing_function(x, **kwargs)

def prepare_data_generators(val_split, batch_size, class_mode, shuffle=True):
  if args.only_luma == True:
    preproc = grayscale_preprocessing_shim
  else:
    preproc = preprocessing_function

  # Prepare data generators
  train_datagen =  ImageDataGenerator(
    #zca_whitening=True,
    preprocessing_function=preproc,
    rotation_range=args.rotation,
    shear_range=args.shear,
    zoom_range=args.zoom,
    horizontal_flip=args.h_flip,
    vertical_flip=args.v_flip,
    validation_split=val_split
  )

  #all_samples = []
  #for catdir in next(os.walk(TRAIN_DIR))[1]:
  #    path = os.path.join(TRAIN_DIR, catdir)
  #    for file in next(os.walk(path))[2]:
  #        if not file.lower().endswith(('.png', '.jpg', '.bmp', '.ppm', '.tif')):
  #            continue
  #        all_samples.append(os.path.join(path, file))
  #rn.shuffle(all_samples)
  #sample_count = int(len(all_samples)*0.01)
  #random_sampling = all_samples[:sample_count]
  #
  #train_data_sample = np.ndarray((len(random_sampling), 64, 64, 3))
  #
  #print("Sampling", len(random_sampling), "images for ZCA whitening.")
  #with progressbar.ProgressBar(max_value=len(random_sampling)) as bar:
  #  for i, image_sample in enumerate(random_sampling):
  #    image = cv2.imread(image_sample,-1)
  #    image = np.float32(cv2.resize(image, (64, 64)))
  #    image = preprocessing_function(image.reshape(64, 64, 3))
  #    train_data_sample[i] = image
  #    bar.update(i)
  #
  #print("Begin data generator fit.")
  #train_datagen.fit(train_data_sample)

  train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                      target_size=(HEIGHT, WIDTH), 
                                                      batch_size=batch_size,
                                                      class_mode=class_mode,
                                                      subset='training',
                                                      shuffle=shuffle)
  if val_split:
    validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                             target_size=(HEIGHT, WIDTH), 
                                                             batch_size=batch_size,
                                                             class_mode=class_mode,
                                                             subset='validation',
                                                             shuffle=shuffle)
  else:
    validation_generator = None

  return train_generator, validation_generator

def build_bottleneck_top_model_vgg16(train_generator, class_list):
  train_data_shape = train_generator[0][0].shape[1:]
  num_classes = len(class_list)

  model = Sequential()
  model.add(Flatten(input_shape=train_data_shape))
  model.add(Dense(1024, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(args.dropout))
  model.add(Dense(num_classes, activation='sigmoid'))

  optimizer = Adam(lr=INIT_LR, decay=INIT_LR / args.num_epochs)

  weights = K.variable([float(1087)/808, float(1087)/1087, float(1087)/601, float(1087)/825, float(1087)/659])
  wcc = weighted_categorical_crossentropy(weights)

  model.compile(optimizer=optimizer, loss=wcc, metrics=['categorical_accuracy'])

  return model

def build_bottleneck_top_model_inceptionv3(inputs, class_list, tensor_only=False):
  #train_data_shape = train_generator[0][0].shape[1:]
  num_classes = len(class_list)

  #####
  #inputs = Input(shape=train_data_shape, name='top_input')
  x = GlobalMaxPooling2D(name='top_pooling')(inputs)
  x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='top_dense_1')(x)
  x = Dropout(args.dropout, name='top_dropout')(x)
  predictions = Dense(num_classes, activation='sigmoid', name='predictions')(x)
  if tensor_only == True:
    return predictions

  model = Model(inputs=inputs, outputs=predictions)
  #####

  #model = Sequential()
  #model.add(GlobalMaxPooling2D(input_shape=train_data_shape))
  #model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.0001)))
  #model.add(Activation('relu'))
  ## model.add(BatchNormalization()) #added
  #model.add(Dropout(args.dropout))
  #model.add(Dense(num_classes, activation='sigmoid'))

  #model.add(Flatten(input_shape=train_data.shape[1:]))
  #model.add(Dense(1024, activation='relu'))
  #model.add(Dropout(args.dropout))
  #model.add(Dense(num_classes, activation='sigmoid'))

  optimizer = SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
  #optimizer = Adam(lr=INIT_LR, decay=INIT_LR / args.num_epochs)
  #optimizer = Adam(lr=INIT_LR)

  weights = K.variable([float(1087)/808, float(1087)/1087, float(1087)/601, float(1087)/825, float(1087)/659])
  wcc = weighted_categorical_crossentropy(weights)

  #model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy', recall, precision])
  model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

  return model

def build_non_bottleneck_top_model(base_model, class_list):
  finetune_model = utils.build_finetune_model(base_model, 
                                              dropout=args.dropout, 
                                              fc_layers=FC_LAYERS, 
                                              num_classes=len(class_list))

  if args.continue_training:
    finetune_model.load_weights(WEIGHTS_PATH)

  adam = Adam(lr=0.00001)
  finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

  return finetune_model

def weighted_crossentropy(target, output, pos_weight, from_logits=False):
  """Weighted crossentropy between an output tensor and a target tensor.
  # Arguments
    target: A tensor with the same shape as `output`.
    output: A tensor.
    from_logits: Whether `output` is expected to be a logits tensor.
        By default, we consider that `output`
        encodes a probability distribution.
  # Returns
    A tensor.
  """
  # Note: tf.nn.weighted_cross_entropy_with_logits
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # transform back to logits
    _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

  return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                  logits=output,
                                                  pos_weight=pos_weight)

def weighted_categorical_crossentropy(weights):
  def loss(y_true, y_pred):
    return weighted_crossentropy(y_pred, y_true, weights)
  return loss

# Prepare the model
if args.model == "VGG16":
  from keras.applications.vgg16 import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "VGG19":
  from keras.applications.vgg19 import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet50":
  from keras.applications.resnet50 import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionV3":
  from keras.applications.inception_v3 import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "Xception":
  from keras.applications.xception import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionResNetV2":
  from keras.application.inception_resnet_v2 import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "MobileNet":
  from keras.applications.mobilenet import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet121":
  from keras.applications.densenet import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet169":
  from keras.applications.densenet import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet201":
  from keras.applications.densenet import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetLarge":
  from keras.applications.nasnet import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = NASNetLarge(weights='imagenet', include_top=True, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetMobile":
  from keras.applications.nasnet import preprocess_input
  preprocessing_function = preprocess_input
  if args.mode != "predict":
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
else:
  ValueError("The model you requested is not supported in Keras")

  
if args.mode == "train":
  print("\n***** Begin training *****")
  print("Dataset -->", DATASET_NAME)
  print("Model -->", args.model)
  print("Resize Height -->", args.resize_height)
  print("Resize Width -->", args.resize_width)
  print("Num Epochs -->", args.num_epochs)
  print("Batch Size -->", args.batch_size)

  print("Data Augmentation:")
  print("\tVertical Flip -->", args.v_flip)
  print("\tHorizontal Flip -->", args.h_flip)
  print("\tRotation -->", args.rotation)
  print("\tZooming -->", args.zoom)
  print("\tShear -->", args.shear)
  print("")

  # If using data augmentation, disable bottlenecks
  if args.v_flip or args.h_flip or args.rotation or args.zoom or args.shear:
    USE_BOTTLENECKS = False

  # Create directories if needed
  if not os.path.isdir("checkpoints"):
    os.makedirs("checkpoints")
  if not os.path.isdir("bottlenecks"):
    os.makedirs("bottlenecks")
  if not os.path.isdir("models"):
    os.makedirs("models")

  if USE_BOTTLENECKS and not args.skip_bottleneck_check:
    save_bottleneck_features()

  class_list = utils.get_subfolders(TRAIN_DIR)
  utils.save_class_list(class_list, model_name=args.model, dataset_name=DATASET_NAME)
  num_classes = len(class_list)

  if(num_classes > 2):
    class_mode = 'categorical'
  else:
    class_mode = 'binary'

  print("Class Mode -->", class_mode, "(" + str(num_classes) + " categories)")

  if USE_BOTTLENECKS:
    train_generator, validation_generator = prepare_bottleneck_data_generators(val_split=args.val_split, batch_size=BATCH_SIZE)
  else:
    train_generator, validation_generator = prepare_data_generators(val_split=args.val_split, batch_size=BATCH_SIZE, class_mode=class_mode)

  tensorboard = TrainValTensorBoard(log_dir='./logs/' + DATASET_NAME + "-{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())), 
                                    histogram_freq=0, 
                                    batch_size=BATCH_SIZE, 
                                    write_graph=False, 
                                    write_grads=False, 
                                    write_images=False, 
                                    embeddings_freq=0, 
                                    embeddings_layer_names=None, 
                                    embeddings_metadata=None, 
                                    embeddings_data=None, 
                                    update_freq='batch')

  def lr_decay(epoch):
    if epoch%20 == 0 and epoch!=0:
      lr = K.get_value(model.optimizer.lr)
      K.set_value(model.optimizer.lr, lr/2)
      print("LR changed to {}".format(lr/2))
    return K.get_value(model.optimizer.lr)

  learning_rate_schedule = LearningRateScheduler(lr_decay)

  if os.path.isfile(WEIGHTS_PATH):
    os.remove(WEIGHTS_PATH)
  if os.path.isfile(MODEL_PATH):
    os.remove(MODEL_PATH)
  if os.path.isfile(FINAL_MODEL_PATH):
    os.remove(FINAL_MODEL_PATH)
  if os.path.isfile(FINAL_WEIGHTS_PATH):
    os.remove(FINAL_WEIGHTS_PATH)

  checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor=["acc"], verbose=1, mode='max', save_weights_only=True)
  callbacks_list = [checkpoint, tensorboard]
  
  if USE_BOTTLENECKS:
    train_data_shape = train_generator[0][0].shape[1:]
    final_model = build_bottleneck_top_model_inceptionv3(Input(shape=train_data_shape, name='top_input'), class_list=class_list)
  else:
    final_model = build_non_bottleneck_top_model(base_model=base_model, class_list=class_list)  

  model_json = final_model.to_json()
  with open(MODEL_PATH, "w") as json_file:
      json_file.write(model_json)
  json_file.close()

  # This is needed because fit_generator dies if steps_per_epoch is < 1 (i.e. batch size is too big)
  # see: https://github.com/keras-team/keras/issues/3657#issuecomment-360522232
  if train_generator.samples < BATCH_SIZE or validation_generator.samples < BATCH_SIZE:
    print("Batch size is too low. Reduce it to less than the number of samples.")
    os._exit(0)

  history = final_model.fit_generator(train_generator, 
                                      epochs=args.num_epochs, 
                                      workers=8, steps_per_epoch=train_generator.samples // BATCH_SIZE, 
                                      validation_data=validation_generator, 
                                      validation_steps=validation_generator.samples // BATCH_SIZE, 
                                      class_weight='auto', 
                                      shuffle=True, 
                                      callbacks=callbacks_list)


  print("Saving final merged model with weights")
  base_inputs = base_model.input
  base_outputs = base_model.output
  outputs = build_bottleneck_top_model_inceptionv3(base_outputs, class_list=class_list, tensor_only=True)
  complete_model = Model(inputs=base_inputs, outputs=outputs)

  #complete_model = utils.merge_topless_top_model(model_bottom=base_model, model_top=final_model)
  model_json = complete_model.to_json()
  with open(FINAL_MODEL_PATH, "w") as json_file:
      json_file.write(model_json)
  json_file.close()

  complete_model.load_weights(WEIGHTS_PATH, by_name=True)
  complete_model.save_weights(FINAL_WEIGHTS_PATH)

  #complete_model.save(FINAL_MODEL_PATH)

  #plot_training(history)

elif args.mode == "predict":

  if args.image is None or args.dataset is None:
    ValueError("You must pass an image path when using prediction mode.")

  images = []

  if args.image_dir is None:
    images.append(args.image)
  else:
    for file in next(os.walk(args.image_dir))[2]:
      if not file.lower().endswith(('.png', '.jpg', '.bmp', '.ppm', '.tif')):
        continue
      images.append(os.path.join(args.image_dir, file))

  images.sort()

  class_list = utils.load_class_list(model_name=args.model, dataset_name=DATASET_NAME)

  #final_model = load_model(FINAL_MODEL_PATH)
  from keras.models import model_from_json
  with open(FINAL_MODEL_PATH, "r") as json_file:
    loaded_model_json = json_file.read()
    final_model = model_from_json(loaded_model_json)
    json_file.close()
  final_model.load_weights(FINAL_WEIGHTS_PATH)

  # Run the classifier and print results
  st = time.time()

  # Read in images
  for file in images:
    print("File: ", file)

    image = cv2.imread(file,-1)
    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))

    out = final_model.predict(image)

    label_weights = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]

    print("Predicted class = ", class_name)
    print("Labels:")
    for index,weight in enumerate(label_weights):
      print("\t", class_list[index], "\t-->", "{0:.2f}%".format(round(weight*100,2)))

  run_time = time.time()-st

  print("Run time = ", run_time)
  #cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)

elif args.mode == "cam":

  if args.image is None:
    ValueError("You must pass an image path when using CAM mode.")

  class_list = utils.load_class_list(model_name=args.model, dataset_name=DATASET_NAME)

  from keras.models import model_from_json
  with open(FINAL_MODEL_PATH, "r") as json_file:
    loaded_model_json = json_file.read()
    final_model = model_from_json(loaded_model_json)
    json_file.close()
  final_model.load_weights(FINAL_WEIGHTS_PATH)

  # Utility to search for layer index by name.
  # Alternatively we can specify this as -1 since it corresponds to the last layer.
  layer_idx = visutils.find_layer_idx(final_model, 'predictions')
  print("Remove Activation from Last Layer")
  # Swap softmax with linear
  final_model.layers[layer_idx].activation = activations.linear
  print("Done. Now Applying changes to the model ...")
  final_model = visutils.apply_modifications(final_model)

  # Run the CAM and print results
  st = time.time()

  # Read in image
  fig = plt.figure(figsize=(10,10))
  img1 = image.load_img(args.image,target_size=(299,299))
  img1 = image.img_to_array(img1)
  img1 = np.expand_dims(img1, axis=0)
  img1 = preprocess_input(img1)
  layer_idx = visutils.find_layer_idx(final_model, 'predictions')
  heatmap = []
  img_init= visutils.load_img(args.image,target_size=(299,299))
  for index in range(len(class_list)):
      heatmap = visualize_cam(final_model, layer_idx, filter_indices=index, seed_input=img1[0,:,:,:])
      ax = fig.add_subplot(int(math.ceil(len(class_list)/3.)), 3, index+1)
      ax.set_title(class_list[index])
      plt.imshow(overlay(img_init, heatmap))
  plt.savefig("./predictions/"+os.path.splitext(os.path.basename(args.image))[0]+".png", dpi=144)

  run_time = time.time()-st

  print("Run time = ", run_time)

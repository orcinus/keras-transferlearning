# Transfer Learning Suite in Keras
Appropriated from https://github.com/GeorgeSeif/Transfer-Learning-Suite and heavily modified.
New:
- bottleneck features for super-fast classifier retraining (within imagenet scope)
- bottleneck generator allows use of large datasets
- automatic on-the-fly splitting into train and validation dataset, with pre-split shuffle
- TensorBoard integration
- precision and recall metrics
- other stuff

Still under heavy construction.

## News

## Description

## Models
All of the Keras built in models are made available:

| Model  | Size  |  Top-1 Accuracy  |  Top-5 Accuracy  |  Parameters  |  Depth  |
| -------------     | ------------- | -------------| ------------- | ------------- | ------------- |
| VGG16    | 528 MB    |  0.715    | 0.901    | 138,357,544    | 23    |
| VGG19    | 549 MB    |  0.727    | 0.910    | 143,667,240    | 26    |
| ResNet50    |  99 MB    | 0.759    | 0.929    | 25,636,712    |  168    |
| Xception    |  88 MB    | 0.790    | 0.945    | 22,910,480     | 126    |
| InceptionV3    | 92 MB    | 0.788    | 0.944    | 23,851,784    |  159    |
| InceptionResNetV2    | 215 MB    |  0.804    | 0.953   | 55,873,736    |  572    |
| MobileNet    | 17 MB    | 0.665    | 0.871    | 4,253,864    | 88    |
| DenseNet121    | 33 MB    | 0.745    | 0.918    | 8,062,504    | 121    |
| DenseNet169    | 57 MB    | 0.759    | 0.928    | 14,307,880    |  169    |
| DenseNet201    | 80 MB    | 0.770    | 0.933    | 20,242,984    |  201    |
| NASNetMobile    | 21 MB    | NA    | NA    | 5,326,716    |  NA    |
| NASNetLarge    | 342 MB    | NA    | NA    | 88,949,818    |  NA    |

    

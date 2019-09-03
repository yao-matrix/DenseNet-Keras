"""Test ImageNet pretrained DenseNet"""

import cv2
import sys
import numpy as np
from keras.optimizers import SGD
import keras.backend as K

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

model_zoo = {"densenet121": "densenet121_weights_tf",
             "densenet161": "densenet161_weights_tf",
             "densenet169": "densenet169_weights_tf"}

## Change target model here to try different models
target_model = 'densenet169'

if target_model == 'densenet121':
    from densenet121 import DenseNet
elif target_model == 'densenet161':
    from densenet161 import DenseNet
elif target_model == 'densenet169':
    from densenet169 import DenseNet
else:
    sys.exit("not supported model")


im = cv2.resize(cv2.imread('resources/shark.jpg'), (224, 224)).astype(np.float32)
# im = cv2.resize(cv2.imread('shark.jpg'), (224, 224)).astype(np.float32)

# Subtract mean pixel and multiple by scaling constant 
# Reference: https://github.com/shicai/DenseNet-Caffe
im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

weights_path = 'imagenet_models/' + model_zoo[target_model] + '.h5'

# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)

K.set_learning_phase(False)

# Test pretrained model
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

output_node_names = ['prob/Softmax']    # Output nodes
session = K.get_session()
graph = session.graph
with graph.as_default():
    optimized_graph_def = graph.as_graph_def()
    optimized_graph_def = tf.graph_util.convert_variables_to_constants(session, optimized_graph_def, output_node_names)
    transforms = ['strip_unused_nodes', 'remove_attribute(attribute_name=_class)', 'fold_constants(ignore_errors=true)',
                  'remove_device', 'remove_nodes(op=Identity, op=CheckNumerics)', "fold_batch_norms", "fold_old_batch_norms"]
    optimized_graph_def = TransformGraph(optimized_graph_def, ["data"], output_node_names, transforms)

    pb_path = 'imagenet_models/' + model_zoo[target_model] + '.pb'
    with open(pb_path, 'wb') as f:
        f.write(optimized_graph_def.SerializeToString())

out = model.predict(im)

# Load ImageNet classes file
classes = []
with open('resources/classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))

print 'Prediction: '+ str(classes[np.argmax(out)])

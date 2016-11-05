# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import caffe

import os

import numpy as np





caffe.set_device(0)

caffe.set_mode_gpu()



model_def = 'models/bvlc_reference_caffenet_utkin/deploy.prototxt'

model_weights = 'models/bvlc_reference_caffenet_utkin/caffenet_utkin_train_iter_11000.caffemodel'



net = caffe.Net(model_def,      # defines the structure of the model

                model_weights,  # contains the trained weights

                caffe.TEST)     # use test mode (e.g., don't perform dropout)



# for testing if python is okay with caffe and net



for layer_name, blob in net.blobs.iteritems():

    print layer_name + '\t' + str(blob.data.shape)





def vis_square(data, filename):

    """Take an array of shape (n, height, width) or (n, height, width, 3)

       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""



    # normalize data for display

    data = (data - data.min()) / (data.max() - data.min())



    # force the number of filters to be square

    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]),

                (0, 1), (0, 1))  # add some space between filters

               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)



    # tile the filters into an image

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    fig = data
    matplotlib.pyplot.savefig("{}.png".format(filename))

    # plt.imshow(data);

    matplotlib.pyplot.axis('off')
    

for layer_name, blob in net.blobs.iteritems():
    filters = net.blobs[layer_name].data[0]
    vis_square(filters.transpose(0, 2, 3, 1), '{}Layer'.format(layer_name))
    
    

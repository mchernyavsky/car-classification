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



net = caffe.Net(model_def, model_weights, caffe.TEST)



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




solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('models/bvlc_reference_caffenet_utkin/solver.prototxt')

[(k, v.data.shape) for k, v in solver.net.blobs.items()]

# [('conv1', (96, 3, 11, 11)),
#  ('conv2', (256, 48, 5, 5)),
#  ('conv3', (384, 256, 3, 3)),
#  ('conv4', (384, 192, 3, 3)),
#  ('conv5', (256, 192, 3, 3)),
#  ('fc6', (4096, 9216)),
#  ('fc7', (4096, 4096)),
#  ('feat', (2, 4096)),
#  ('conv1_p', (96, 3, 11, 11)),
#  ('conv2_p', (256, 48, 5, 5)),
#  ('conv3_p', (384, 256, 3, 3)),
#  ('conv4_p', (384, 192, 3, 3)),
#  ('conv5_p', (256, 192, 3, 3)),
#  ('fc6_p', (4096, 9216)),
#  ('fc7_p', (4096, 4096)),
#  ('feat_p', (2, 4096))]



solver.net.forward()
solver.test_nets[0].forward()

plt.imshow(solver.net.blobs['data'].data[:64, 0].transpose(1, 0, 2).reshape(227, 64*227), cmap='gray');
plt.axis('off')

solver.step(1)


plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(96, 3, 11, 11).transpose(0, 2, 1, 3).reshape(96*3, 11*11), cmap='gray');
plt.axis('off')
plt.savefig("conv1.png")



print 'test labels:', solver.test_nets[0].blobs['sim'].data[:256]

niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['sim'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()


# python /pathtocaffe/tools/extra/parse_log.py lenet_train.log .



    
# plotting loss


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import caffe
from sklearn import preprocessing

train_log = pd.read_csv("./proper_train.log.train")
test_log = pd.read_csv("./proper_train.log.test")
_, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
plt.savefig("PlotLoss.png")
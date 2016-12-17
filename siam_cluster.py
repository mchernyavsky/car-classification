# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import caffe

MODEL_FILE = "models/bvlc_reference_caffenet_utkin/train_val_siam.prototxt"
PRETRAINED_FILE = "models/bvlc_reference_caffenet_utkin/caffenet_utkin_train_iter_3074.caffemodel"

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
    

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import caffe

MODEL_FILE = "models/bvlc_reference_caffenet_utkin/train_val_siam.prototxt"
PRETRAINED_FILE = "models/bvlc_reference_caffenet_utkin/caffenet_utkin_train_iter_500.caffemodel"

caffe.set_device(0)
caffe.set_mode_gpu()
meanFile = 'datasets/utkin/data/utkin_mean.npy'
image_list_file = 'datasets/utkin/data/train.txt'

imageBasePath = 'datasets/utkin/data/train/'
postfix = 'feat.txt'
 
def readImageList(imageListFile):
    imageList = []
    labels = []
    limit = 1000
    with open(imageListFile,'r') as fi:
        line_num = 0
        while(line_num < limit):
            line_num += 1
            line = fi.readline().strip().split()
            if not line:
                break
            imageList.append(line[0])
            labels.append(line[1]) 
    print 'read imageList done image num ', len(imageList)
    return imageList, labels
    
data, labels = readImageList(image_list_file)
MEAN_PROTO_PATH = '/datasets/utkin/data/utkin_mean.binaryproto'
MEAN_NPY_PATH = '/datasets/utkin/data/utkin_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
MEAN_PROTO_PATH = '/opt/alexander/datasets/utkin/data/utkin_mean.binaryproto'
MEAN_NPY_PATH = '/opt/alexander/datasets/utkin/data/utkin_mean.npy' 
data = open(MEAN_PROTO_PATH, 'rb' ).read()  
blob.ParseFromString(data)
array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
np.save(MEAN_NPY_PATH ,mean_npy)
meanFile = 'datasets/utkin/data/utkin_mean.npy'
image_list_file = 'datasets/utkin/data/train1.txt'

imageBasePath = 'datasets/utkin/data/train1/'
postfix = 'feat.txt'
 
def readImageList(imageListFile):
    imageList = []
    labels = []
    limit = 1000
    with open(imageListFile,'r') as fi:
        line_num = 0
        while(line_num < limit):
            line_num += 1
            line = fi.readline().strip().split()
            if not line:
                break
            imageList.append(line[0])
            labels.append(line[1]) 
    print 'read imageList done image num ', len(imageList)
    return imageList, labels
    
data, labels = readImageList(image_list_file)
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)

def extractFeature(imageList, net):
    y = []
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(meanFile).mean(1).mean(1)) 
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    net.blobs['data'].reshape(1,3,227,227) 
    num=0
    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile)
        print imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
        y.append(net.blobs["feat"].data[0])
        num +=1
    return np.array(y)
    
    
feat = extractFeature(data, net)
feat.shape
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
feat = extractFeature(data, net)
feat.shape
np.unique(labels)
def show_clusters(data, y, name):
    X = data
    print "X.shape = ", X.shape
    print "y.shape = ", y.shape
    cm = plt.cm.get_cmap('Accent')
    plt.scatter(X[:,0], X[:,1], c=y, vmin=0, vmax=16, cmap=cm)
    plt.axis('off')
    plt.show()
    plt.savefig("Clusters{}.png".format(name), dpi=600)
    
show_clusters(feat, labels, "Train")

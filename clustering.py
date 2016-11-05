# coding: utf-8
import matplotlib
matplotlib.use('Agg')  # for running in container in remote host
import matplotlib.pyplot as plt
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
    plt.imshow(data);
    plt.savefig("{}.png".format(filename), dpi=1200)
    plt.axis('off')
 
# extracting features to txt files for 1000 images

meanFile = 'datasets/utkin/data/utkin_mean.npy'
image_list_file = 'datasets/utkin/data/train.txt'

imageBasePath = 'datasets/utkin/data/train/'
postfix = 'fc7.txt'
 
def readImageList(imageListFile):
    imageList = []
    limit = 2000
    with open(imageListFile,'r') as fi:
        line_num = 0
        while(line_num < limit):
            line_num += 1
            line = fi.readline().strip().split()
            if not line:
                break
            imageList.append(line[0]) 
    print 'read imageList done image num ', len(imageList)
    return imageList[1000:]


from sklearn.manifold import TSNE
from scipy import cluster


def get_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            features = line.strip().split(' ')
            data.append(features)
    return np.array(data)


   
    


y = [float(e)/max(y) for e in y]
def show_clusters(data, y):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = model.fit_transform(data[1:])
    print "X.shape = ", X.shape
    print "y.shape = ", y.shape
    cm = plt.cm.get_cmap('Accent')
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cm)
    plt.axis('off')
    plt.show()
    plt.savefig("Clusters.png", dpi=600)
    
    
    
show_clusters(data, y)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels = le.fit(y)
labels
labels.classes_
labels = le.transform(y)
labels
data = get_data("datasets/utkin/data/train/fc7_feature_txt_total.txt")
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
        y.append(out['prob'][0].argmax())
        # fea_file = imagefile_abs.replace('.jpg',postfix)
        num +=1
        # print 'Num ',num,' extract feature ',fea_file
        # np.savetxt(fea_file, net.blobs['fc7'].data)
    return y
y = extractFeature(images_list, net)
    
  
    
show_clusters(X, labs)
def show_clusters(data, y):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = model.fit_transform(data[:, 1:])
    print "X.shape = ", X.shape
    print "y.shape = ", y.shape
    cm = plt.cm.get_cmap('Accent')
    plt.scatter(X[:,0], X[:,1], c=y, vmin=0, vmax=5, cmap=cm)
    plt.axis('off')
    plt.show()
    plt.savefig("Clusters.png", dpi=600)
    

X = []
Y = []
for k, g in groupby(xy, lambda x: x[1]):
     p = list(g)
     X += p
     Y += [k] * len(p)
     

    

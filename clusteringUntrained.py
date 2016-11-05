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
 
meanFile = 'datasets/utkin/data/utkin_mean.npy'
image_list_file = 'untrained.txt'
imageBasePath = 'untrained/'
def extractFeature(imageList, net):
    y = []
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(meanFile).mean(1).mean(1)) 
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    net.blobs['data'].reshape(1,3,227,227) 
    num=0
    features = []
    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile)
        print imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
        y.append(out['prob'][0])
        fea_file = imagefile_abs.replace('.jpg',postfix)
        num +=1
        print 'Num ',num,' extract feature ',fea_file
        features.append(net.blobs['fc7'].data)
        # np.savetxt(fea_file, net.blobs['fc7'].data)
    return features
    
def readImageList(imageListFile):
    imageList = []
    limit = 2000
    labels = []
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
    
data, labs = readImageList(image_list_file)
untrainedFeatures = extractFeature(image_list_file, net)

def extractFeature(imageList, net):
    y = []
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(meanFile).mean(1).mean(1)) 
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    net.blobs['data'].reshape(1,3,227,227) 
    num=0
    features = []
    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile)
        print imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
        # y.append(out['prob'][0])
        #fea_file = imagefile_abs.replace('.jpg',postfix)
        num +=1
        print 'Num ',num,' extract feature '
        features.append(net.blobs['fc7'].data)
        # np.savetxt(fea_file, net.blobs['fc7'].data)
    return features
    
untrainedFeatures = extractFeature(data, net)
untrainedFeatures
untrainedFeatures.shape
untrainedFeatures = np.array(untrainedFeatures)
untrainedFeatures.shape
untrainedFeatures[0]
untrainedFeatures[0][0]
for i in range(len(untrainedFeatures)):
   untrainedFeatures[i] = untrainedFeatures[i][0]
   
untrainedFeatures.shape
untrainedFeatures[0]
for i in range(len(untrainedFeatures)):
   untrainedFeatures[i] = untrainedFeatures[i][0]
   
   
untrainedFeatures.shape
untrainedFeatures[0]
untrainedFeatures[0][0]
untrainedFeatures[0][0][0]
untrainedFeatures[0].shape
untrainedFeatures.shape
untrainedFeatures[0]
untrainedFeatures[0][0]
untrainedFeatures[2][0]

featureTxtListFile = 'datasets/utkin/data/train.txt'
featureTxtBasePath = 'datasets/utkin/data/train/'
featureTxtStoreBasePath = 'datasets/utkin/data/train/'
postfix = ''

def narray_vstack(featuretxtList):
    total_array = np.arange(4097) # initial array, will be deleted later
    num = 0
    for featuretxtfile in featuretxtList:
        featuretxtfile_abs = os.path.join(featureTxtBasePath,featuretxtfile)
        #print featuretxtfile_abs
        #imagenumber = .replace('fc7.txt', postfix)
        #imagenumber = float(int(featuretxtfile.replace('fc7.txt', postfix)))
        #print imagenumber

        temp_array = np.loadtxt(featuretxtfile_abs)
        temp_array = np.append([num], temp_array)
        total_array = np.vstack((total_array, temp_array))
        num += 1
        print 'loaded num ', num - 1
        if num % 500 == 0:
            print 'txt loaded number:', num
    np.savetxt(os.path.join( featureTxtStoreBasePath,'fc7_feature_txt_total.txt'), total_array)


def readFeatureTxtList(featureTxtListFile):
    featuretxtList = []
    limit = 2000
    with open(featureTxtListFile,'r') as fi:
        while(True):
            line = fi.readline().strip().split()# every line is a image file name
            if not line:
                break
            name = line[0].replace(".jpg", 'fc7.txt')
            featuretxtList.append(name) 
    print 'read feattxtList done featuretxt num ', len(featuretxtList)
    return featuretxtList[1000:2000]


featuretxtList = readFeatureTxtList(featureTxtListFile)
narray_vstack(featuretxtList)
from sklearn.manifold import TSNE
from scipy import cluster

def get_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            features = line.strip().split(' ')
            data.append(features)
    return np.array(data)
data = get_data("datasets/utkin/data/train/fc7_feature_txt_total.txt")


data = data[:-1]
data = data[:,1:]
cols = ['b'] * 1000

for i in labs:
    if int(i) == 19:
        cols.append('r')
    else:
        cols.append('g')
     

for i in untrainedFeatures:
    data.append(i[0])
    

for i in range(len(untrainedFeatures)):
    untrainedFeatures[i] =  untrainedFeatures[i].flatten()

    
#for i in untrainedFeatures:
#    np.append(data, i[0])
    
#data.shape
#untrainedFeatures.shape
untrFeat = untrainedFeatures[:, 0]
untrFeat.shape
X = []


    
X = np.concatenate((data,untrFeat), axis=0)


cols
for i in range(len(cols)):
    if cols[i] == 'b':
        cols[i] = 'black'
    elif cols[i] == 'r':
        cols[i] = 'blue'
    else:
        cols[i] = 'yellow'

for a,b in zip(X, cols):
    file.write(str(a) + " " + str(b) + "\n")
        
with open("clustersSave.txt", "w+") as file:
    for a,b in zip(X, cols):
        file.write(str(a) + " " + str(b) + "\n")
        
def show_clusters(data, y, name, params=None):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    if params is not None:
        model.set_params(**params)
    X = model.fit_transform(data)
    print X
    p = model.get_params()
    print "X.shape = ", X.shape
    print "y.shape = ", y.shape
    print y
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.gray()
    plt.axis('off')
    plt.show()
    plt.savefig("ClustersUntrained{}.png".format(name), dpi=600)
    plt.clf()
    return p
show_clusters(X[:1100], np.array(cols[:1100]), 'Blue', params=p)
show_clusters(X_yel, np.array(cols_yel), 'Yellow')


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn.manifold import TSNE
import caffe
import numpy as np
from sklearn import preprocessing

# directory with models from docker container
PATH = '/opt/alexander/models/bvlc_reference_caffenet_utkin/'
ZERO = '0'
CAFFE_MODEL = 'caffemodel'
MODEL_FILE = "models/bvlc_reference_caffenet_utkin/train_val_siam.prototxt"
PRETRAINED_FILE = "models/bvlc_reference_caffenet_utkin/caffenet_utkin_train_iter_44000.caffemodel"
PLOTS = "/opt/alexander/plots/"


models = [f for f in listdir(PATH) if isfile(join(PATH, f))]

# only models for siamese (with two zeros and only caffemodel files we need)
models = [model for model in models if (ZERO + ZERO + ZERO) in model and CAFFE_MODEL in model]
models = [model for model in models if (ZERO + ZERO + ZERO + ZERO) in model and CAFFE_MODEL in model]
models.append('caffenet_utkin_train_iter_44000.caffemodel')


# setting up caffe
caffe.set_device(0)
caffe.set_mode_gpu()


# get all layers
#net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
#for layer_name, blob in net.blobs.iteritems():
#    print layer_name + '\t' + str(blob.data.shape)

# so we have only two last full-connected layers, so will be working on them

layers = ['fc6', 'fc7']


# fetching data for learn and test plotting
train_file = 'datasets/utkin/data/train1.txt'
train_folder = 'datasets/utkin/data/train1/'
val_file = 'datasets/utkin/data/val1.txt'
val_folder = 'datasets/utkin/data/val1/'


def read_images(imageListFile):
    imageList = []
    labels = []
    limit = 1000
    with open(imageListFile, 'r') as fi:
        line_num = 0
        while line_num < limit:
            line_num += 1
            line = fi.readline().strip().split()
            if not line:
                break
            imageList.append(line[0])
            labels.append(line[1])
    print 'Done reading images: ', len(imageList)
    return imageList, labels


data_train, labels_train = read_images(train_file)
data_val, labels_val = read_images(val_file)


def extract_features(images, image_folder, net, layer):
    y = []
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_raw_scale('data', 255)
    net.blobs['data'].reshape(1, 227, 227)
    num = 0
    for image_file in images:
        imagefile_abs = join(image_folder, image_file)
        print imagefile_abs
        transformed = transformer.preprocess('data', caffe.io.load_image(imagefile_abs, color=False))
        transformed = np.transpose(transformed, (2, 0, 1))
        net.blobs['data'].data[...] = transformed
        out = net.forward()
        y.append(net.blobs[layer].data[0].copy())
        num += 1
    return np.array(y)


def show_clusters(data, y, name):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(data)
    print "X.shape = ", X.shape
    print "y.shape = ", y.shape
    cm = plt.cm.get_cmap('Accent')
    plt.scatter(X[:, 0], X[:, 1], c=y, vmin=min(y), vmax=max(y), cmap=cm)
    plt.axis('off')
    plt.show()
    plt.savefig(name)
    plt.clf()

for model in models:
    caffe_model = PATH + model
    net_iter = caffe.Net(MODEL_FILE, caffe_model, caffe.TEST)
    print 'Network initialized'
    for layer in layers:
        feat_train = extract_features(data_train, train_folder, net_iter, layer)
        feat_val = extract_features(data_val, val_folder, net_iter, layer)
        labels_train = preprocessing.LabelEncoder().fit_transform(labels_train)
        labels_val = preprocessing.LabelEncoder().fit_transform(labels_val)
        iters = model.split('.')[0]
        name = "{}.png".format(iters)
        show_clusters(feat_train, labels_train, PLOTS + layer + '/' + 'train/' + name)
        show_clusters(feat_val, labels_val, PLOTS + layer + '/' + 'val/' + name)




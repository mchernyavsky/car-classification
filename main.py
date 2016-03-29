import glob

import numpy as np

from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import rectify, softmax
from nolearn.lasagne import NeuralNet

from skimage import transform, io
from skimage.feature import hog

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib
from sklearn.utils import shuffle


def load_data():
    models = ["honda_civic", "hyundai_accent", "hyundai_solaris", "kia_ceed",
              "kia_rio",  "lada_kalina", "mazda_3", "nissan_qashqai",
              "opel_astra", "toyota_camry"]

    X, y = [], []

    for i, model in enumerate(models):
        for img_name in glob.iglob("dataset/cropped/{}/*".format(model)):
            img = io.imread(fname=img_name, as_grey=True)
            img = transform.resize(image=img, output_shape=(192, 240))
            X.append(img)
            y.append(i)

    X = np.array(X).astype(np.float32)
    X = X.reshape((-1, 1, 192, 240))
    y = np.array(y).astype(np.int8)

    X, y = shuffle(X, y, random_state=42)

    return X, y


def rf(X, y):
    X = np.array([hog(x, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
                  for x in X[:, 0]]).astype(np.float32)
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
    rf_clf_scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv)
    print(rf_clf_scores.mean())


def cnn(X, y):
    net = NeuralNet(
        layers=[
                ('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],

        # input layer
        input_shape=(None, 1, 192, 240),

        # layer conv2d1
        conv2d1_num_filters=16,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=rectify,

        # layer maxpool1
        maxpool1_pool_size=(2, 2),

        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(2, 2),
        conv2d2_nonlinearity=rectify,

        # layer maxpool2
        maxpool2_pool_size=(2, 2),

        # dropout1
        dropout1_p=0.5,

        # dense
        dense_num_units=256,
        dense_nonlinearity=rectify,

        # dropout2
        dropout2_p=0.5,

        # output
        output_nonlinearity=softmax,
        output_num_units=10,

        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=16,
        verbose=1,
    )
    net.fit(X, y)

    joblib.dump(net, "simple_cnn.pkl", compress=3)


def main():
    X, y = load_data()
    cnn(X, y)


if __name__ == "__main__":
    main()

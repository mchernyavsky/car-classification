import numpy as np
import re
from skimage import transform, io
from skimage.feature import hog
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from data_preprocessing import preprocess_utkin


def rf(X, y):
    X = np.array([hog(x, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
                  for x in X[:, 0]]).astype(np.float32)

    clf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    cv = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
    rf_clf_scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    print(rf_clf_scores.mean())


def main():
    df = preprocess_utkin()

    X = []
    for img_name in df["photo_path"]:
        img_name_old = re.sub("\.jpg", "_fr.jpg", img_name)
        img = io.imread("datasets/utkin/src/utkin-download/" + img_name_old,
                        as_grey=True)
        img = img[:480, :640]
        img = transform.resize(img, output_shape=(192, 240))
        X.append(img)

    X = np.array(X).astype(np.float32)
    X = X.reshape((-1, 1, 192, 240))
    y = np.array(df["label"]).astype(np.int32)
    X, y = shuffle(X, y, random_state=42)

    rf(X, y)


if __name__ == "__main__":
    main()


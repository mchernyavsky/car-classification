import glob

from skimage import transform, io
from skimage.feature import hog

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier


def main():
    models = ["honda_civic", "hyundai_accent", "hyundai_solaris", "kia_ceed",
              "kia_rio",  "lada_kalina", "mazda_3", "nissan_qashqai",
              "opel_astra", "toyota_camry"]

    X, y = [], []

    for i, model in enumerate(models):
        for img_name in glob.iglob("dataset/cropped/{}/*".format(model)):
            img = io.imread(fname=img_name, as_grey=True)

            # output_shape=(64, 80)
            # output_shape=(128, 160)
            # output_shape=(192, 240)
            # output_shape=(256, 320)
            # output_shape=(320, 400)
            img = transform.resize(image=img, output_shape=(192, 240))

            # pixels_per_cell=(8, 8)
            # pixels_per_cell=(16, 16)
            # pixels_per_cell=(32, 32)
            # cells_per_block=(1, 1)
            # cells_per_block=(2, 2)
            # cells_per_block=(3, 3)
            # cells_per_block=(4, 5)
            fd = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

            X.append(fd)
            y.append(i)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
    rf_clf_scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv)
    print(rf_clf_scores.mean())


if __name__ == "__main__":
    main()

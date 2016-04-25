import numpy as np
import pandas as pd
import re
from datetime import time
from skimage import transform, io
from sklearn.preprocessing import LabelEncoder


def preprocess_utkin():
    df = pd.read_excel("datasets/utkin/src/Radar_data_recognized.xlsx")

    df = df.loc[(df["Время"] > time(7)) & (df["Время"] < time(19))]
    df = df.groupby("Название микросегмента").filter(lambda x: len(x) > 9)
    print(len(df), "rows")

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df["Название микросегмента"])
    print(df["label"].max(), "labels")

    df = df[["photo_path", "label"]].iloc[np.random.permutation(len(df))]

    train_val_board = 900
    df.iloc[:train_val_board].to_csv("datasets/utkin/data/train.txt",
                                     sep=" ", header=False, index=False)
    df.iloc[train_val_board:].to_csv("datasets/utkin/data/val.txt",
                                     sep=" ", header=False, index=False)

    for img_path in df.iloc[:train_val_board, 0]:
        img_path_old = re.sub("\.jpg", "_fr.jpg", img_path)
        img = io.imread("datasets/utkin/src/utkin-download/" +
                        img_path_old, as_grey=True)
        img = img[:480, :640]
        img = transform.resize(img, output_shape=(256, 256))
        io.imsave("datasets/utkin/data/train/" + img_path, img)

    for img_path in df.iloc[train_val_board:, 0]:
        img_path_old = re.sub("\.jpg", "_fr.jpg", img_path)
        img = io.imread("datasets/utkin/src/utkin-download/" +
                        img_path_old, as_grey=True)
        img = img[:480, :640]
        img = transform.resize(img, output_shape=(256, 256))
        io.imsave("datasets/utkin/data/val/" + img_path, img)

    return df


def preprocess_avito():
    train = pd.read_csv("datasets/avito/src/avito_2016_train_labels.csv")
    train.image_id = train.image_id.apply(lambda x: "{}.jpg".format(x))
    train.to_csv("datasets/avito/data/train.txt",
                 sep=" ", header=False, index=False)

    test = pd.read_csv("datasets/avito/src/avito_2016_test_labels.csv")
    test.image_id = test.image_id.apply(lambda x: "{}.jpg".format(x))
    test.to_csv("datasets/avito/data/val.txt",
                sep=" ", header=False, index=False)

    print(train.label.max())

if __name__ == "__main__":
    preprocess_utkin()

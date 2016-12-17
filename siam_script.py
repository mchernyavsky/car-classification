import csv
import os
from random import shuffle
# import paramiko
# import pandas as pd
# import scipy as sp
# import pipes
# import subprocess
from itertools import groupby, combinations
from shutil import copyfile
# from sklearn.cross_validation import train_test_split


def read_images(filename=None):
    """
    Reading list of images from file
    :param filename: name of csv file, that contains path to files
    :return: list with elements following format : [image_name, image_path, microsegment, prob]
    """
    images_list = []
    with open(filename, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
	    #if len(images_list) > 5000:
	    #break
	    # print row
	    try:
            	plate, link, microsegment, prob = row
	    except ValueError as e:
		print row
            image_name = link.split('/')[-1]
            image_path = '/'.join(link.split('/')[-2:])
            car = [image_name, image_path, microsegment, prob]
            if len(link) != 0:
                #  print "Image exists and added"
		images_list.append(car)
    return images_list


def select_images(img_list, n):
    """
    Select images which have enough volume in population
    :param img_list: list of images to select
    :return: list of selected images
    """
    limit = 100
    number_of_classes = 0
    res = dict()
    labels = dict()
    last = []
    images_list = sorted(img_list, key=lambda x: int(x[2]))
    for k, g in groupby(images_list, lambda x: int(x[2])):
	#print k
	# print k
        if len(res) >= limit:
            break
	#m = list(g)
        m = []
	for i in list(g):
            if len(m) >= n:
	        break
	    if os.path.exists("/home/synaps/images/{}".format(i[1])):
		# print "exists"
		m.append(i)
        # m = filter(lambda x: os.path.exists("/home/synaps/images/{}".format(x[1])), list(g))[:400]
	# print len(m)
        if len(m) >= n:
	    labels[k] = number_of_classes
	    print "label encoded = ", labels[k], " ", k 
	    number_of_classes += 1
	    res[k] = m[:n]
            last.append(k)
    print "not in learn : ", last[-2:]
    res = {k: v for k, v in res.iteritems() if k not in last[-2:]}
    print "number of classes = ", number_of_classes - 2
    return res, labels


def save_images(data1, data2,  y1, y2,  folder_name):
    folder_path = "./datasets/utkin/data/{}1/".format(folder_name)
    storage_path = "/home/synaps/images/{}"
    n1 = len(data1)
    n2 = len(data2)
    # size = len(data) // 2
    file = open("./datasets/utkin/data/" + folder_name + "1.txt", "w+")
    for img,l in zip(data1, y1):
        file.write(img.split('/')[-1] + " " + str(l)+"\n")
    file.close()
    for img in data1:
        img_path = storage_path.format(img)
	# print img_path
        copyfile(img_path, folder_path + img_path.split('/')[-1])
    folder_path = "./datasets/utkin/data/{}2/".format(folder_name)
    file1 = open("./datasets/utkin/data/" + folder_name + "2.txt", "w+")
    for img,l in zip(data2, y2):
        file1.write(img.split('/')[-1] + " " + str(l)+"\n")
    file1.close()
    for img in data2:
        img_path = storage_path.format(img)
        # print img_path
        copyfile(img_path, folder_path + img_path.split('/')[-1])



def cross_validation(data, labels,  n, test_size=0.25):
    # X, y = [], []
    # random.shuffle(data)
    # n = len(data)
    m = int((1 - test_size) * n) 
    tr_half = m / 2
    val_half = (n - m) / 2 
    X_learn1, X_learn2, y_learn1, y_learn2 = [], [], [], []
    X_validate1, X_validate2,  y_validate1, y_validate2 = [], [], [], []
    cars = []
    for k, g in data.items():
        for item in g:
            cars.append((item[1], labels[int(item[2])]))
    pairs_sim = []
    pairs_not_sim = []
    for img1, img2 in combinations(cars, 2):
        if img1[1] == img2[1]:
            pairs_sim.append((img1, img2))
        else:
            pairs_not_sim.append((img1, img2))
    size = len(pairs_sim)
    pairs_not_sim = pairs_not_sim[:size]
    border = int((1 - test_size) * 2 * size)
    num_meas = 0
    for p1, p2 in zip(pairs_sim, pairs_not_sim):
        for p in (p1, p2):
            img1, img2 = p
            path1, lab1 = img1
            path2, lab2 = img2
            if num_meas <= border:
                X_learn1.append(path1)
                X_learn2.append(path2)
                y_learn1.append(lab1)
                y_learn2.append(lab2)
            else:
                X_validate1.append(path1)
                X_validate2.append(path2)
                y_validate1.append(lab1)
                y_validate2.append(lab2)
            num_meas += 1
    train1 = zip(X_learn1, y_learn1)
    train2 = zip(X_learn2, y_learn2)
    train = zip(train1, train2)
    val1 = zip(X_validate1, y_validate1)
    val2 = zip(X_validate2, y_validate2)
    val = zip(val1, val2)
    shuffle(train)
    shuffle(val)
    train1, train2 = zip(*train)
    val1, val2 = zip(*val)
    X_learn1, y_learn1 = zip(*train1)
    X_learn2, y_learn2 = zip(*train2)
    X_validate1, y_validate1 = zip(*val1)
    X_validate2, y_validate2 = zip(*val2)
    # for k, g in data.items():
	 #    shuffle(g)
    #     for e in g[:tr_half]:
	 #        X_learn1.append(e[1])
    #         y_learn1.append(labels[int(e[2])])
	 #    for e in g[tr_half:m]:
	 #        X_learn2.append(e[1])
	 #        y_learn2.append(labels[int(e[2])])
	 #    for e in g[m: m + val_half]:
	 #        X_validate1.append(e[1])
	 #        y_validate1.append(labels[int(e[2])])
    #     for e in g[m + val_half:n]:
	 #    X_validate2.append(e[1])
    #         y_validate2.append(labels[int(e[2])])
    #for car in data:
    #    X.append(car[1])
    #    y.append(car[2])
    #X_learn, X_validate, y_learn, y_validate = X[:m], X[m:], y[:m], y[m:]
    save_images(data1=X_learn1, data2=X_learn2,y1=y_learn1, y2=y_learn2, folder_name="train")
    save_images(data1=X_validate1, data2=X_validate2, y1=y_validate1, y2=y_validate2, folder_name="val")


images_list = read_images(filename='Microsegments_data.csv')
print "images_list len =",  len(images_list)
num_of_examples = 60
selected_imgs, labels = select_images(images_list, num_of_examples)
print "number of selected classes = ", len(selected_imgs)
#print selected_imgs[0]
cross_validation(selected_imgs, labels, num_of_examples)

# !scripts/create_net.sh utkin
# !scripts/make_net_mean.sh utkin
# !$CAFFE_ROOT/build/tools/caffe train --solver=models/bvlc_reference_caffenet_utkin/solver.prototxt



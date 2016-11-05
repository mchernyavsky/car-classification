import csv
import os
import random
# import paramiko
# import pandas as pd
# import scipy as sp
# import pipes
# import subprocess
from itertools import groupby
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


def file_exists(path):
    local_path = "/home/synaps/images/{}".format(path)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    _, stdout, _ = client.exec_command("[ -f {} ] && echo OK".format(local_path))
    if "OK" in stdout.read():
        return True
    return False


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
    images_list = sorted(img_list, key=lambda x: int(x[2]))
    for k, g in groupby(images_list, lambda x: int(x[2])):
	# print k
        if len(res) >= limit:
            break
	#m = list(g)
        m = []
	for i in g:
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
    print "number of classes = ", number_of_classes
    return res, labels


def save_images(data, y, folder_name):
    folder_path = "./datasets/utkin/data/{}/".format(folder_name)
    storage_path = "/home/synaps/images/{}"
    file = open("./datasets/utkin/data/" + folder_name + ".txt", "w+")
    for img,l in zip(data, y):
        file.write(img.split('/')[-1] + " " + str(l)+"\n")
    file.close()
    for img in data:
        img_path = storage_path.format(img)
	# print img_path
        copyfile(img_path, folder_path + img_path.split('/')[-1])


def cross_validation(data, labels,  n, test_size=0.25):
    # X, y = [], []
    # random.shuffle(data)
    # n = len(data)
    m = int((1 - test_size) * n)
    X_learn, y_learn = [], []
    X_validate, y_validate = [], []
    for k, g in data.items():
	random.shuffle(g)
	for e in g[:m]:
	    X_learn.append(e[1])
	    y_learn.append(labels[int(e[2])])
	for e in g[m:n]:
	    X_validate.append(e[1])
	    y_validate.append(labels[int(e[2])])
    #for car in data:
    #    X.append(car[1])
    #    y.append(car[2])
    #X_learn, X_validate, y_learn, y_validate = X[:m], X[m:], y[:m], y[m:]
    save_images(data=X_learn,y=y_learn, folder_name="train")
    save_images(data=X_validate,y=y_validate, folder_name="val")


images_list = read_images(filename='Microsegments_data.csv')
print "images_list len =",  len(images_list)
num_of_examples = 300
selected_imgs, labels = select_images(images_list, num_of_examples)
print "number of selected classes = ", len(selected_imgs)
#print selected_imgs[0]
cross_validation(selected_imgs, labels, num_of_examples)

# !scripts/create_net.sh utkin
# !scripts/make_net_mean.sh utkin
# !$CAFFE_ROOT/build/tools/caffe train --solver=models/bvlc_reference_caffenet_utkin/solver.prototxt


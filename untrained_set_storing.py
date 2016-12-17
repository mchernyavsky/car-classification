import csv
from itertools import groupby
import os


def read_images_untrained(filename=None, trained=None):
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
            if len(link) != 0 and int(microsegment) not in trained:
                #  print "Image exists and added"
                images_list.append(car)
    return images_list


def get_trained_classes():
    trained = []
    with open("data.txt", 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                seg = str(row[0]).split(' ')[-1] # seg for microsegment
            except ValueError as e:
                print row
            trained.append(int(seg))
    return set(trained)


def select_images(img_list, n):
    """
    Select images which have enough volume in population
    :param img_list: list of images to select
    :return: list of selected images
    """
    limit = 2
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


trained = get_trained_classes()
print trained
images_list = read_images_untrained(filename='Microsegments data.csv', trained=trained)
print "images_list len =",  len(images_list)
untrained = select_images(images_list, 100)
print len(untrained)

with open("untrained.txt", "w+") as file:
    for i in untrained:
        file.writeln(i[1], i[2])


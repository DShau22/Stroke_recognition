import hyperparameters
import random
import os
import numpy as np
import glob
import tensorflow as tf
from math import cos, sin
#creates a big ass list: each entry is a pair, first is label, second is acc data, pop it while training
#For now, just segments a sample data text file into strokes
hparams = hyperparameters.hparams

#rotates axes of measurement by rotation_angle as defined in hparams and
#stores the rotated data into aug_data folder
def augment_data(hparams):
    #os.chdir(hparams.data_dir)
    for filename in glob.glob(hparams.data_dir + '*.txt'):
        with open(filename, 'r') as raw:
            with open(filename[:len(filename) - 4] + "_aug.txt", "w+") as aug:
                for raw_line in raw:
                    accs_data = raw_line.strip('\n').split(',')
                    acc_data = cast_and_fix_bad_data(accs_data)
                    aug_data = [acc_data[0], #z
                                acc_data[1] * cos(hparams.rotation_angle) + acc_data[2] * sin(hparams.rotation_angle), #y
                                acc_data[2] * cos(hparams.rotation_angle) - acc_data[1] * sin(hparams.rotation_angle)] #x
                    aug.write("%d,%d,%d\n" % (aug_data[0], aug_data[1], aug_data[2])) #z, y, x

#accounts for corner cases in wearable where acc prints "-" or nothing,
#and returns data in form or ints
def cast_and_fix_bad_data(accs):
    if any(["-" == el for el in accs]):
        for i in range(len(accs)):
            try:
                accs[accs.index("-")] = 0
            except ValueError:
                pass
    while len(accs) < 3:
        accs.append(0)
    return [int(el) for el in accs]

#puts data from text file to numpy array of shape (3, filelength - 600)
def allocate_data(text_file_path):
    #text file has x, y, z accs in "z,y,x\n" format for every .1 secs
    total_accs = [[],[],[]]
    with open(text_file_path, 'r') as f:
        #take out first minute
        for i in range(600):
            f.readline()
        line_count = 0
        for line in f:
            accs = line.strip('\n').split(',')
            accs = np.array(cast_and_fix_bad_data(accs))
            #accs should now be arr of length 3 with integer entries
            total_accs[0].append(accs[0])
            total_accs[1].append(accs[1])
            total_accs[2].append(accs[2])
            line_count += 1
    return total_accs, line_count

#splits data from numpy array into batch_size number of random segments of shape (3, window size)
def make_batches(hparams, pair_input):
    num_measurements = len(pair_input[1][0]) #number of accerelation measurements per axis
    data_batch, label_batch = [], []
    for i in range(hparams.batch_size):
        random_index = random.randint(0, num_measurements - hparams.batch_size)
        #this transposes the arr to allow it to fit the shape of the placeholder
        data_to_add = list(zip(pair_input[1][0][random_index:random_index + hparams.batch_size],
                               pair_input[1][1][random_index:random_index + hparams.batch_size],
                               pair_input[1][2][random_index:random_index + hparams.batch_size],))
        labels_to_add = pair_input[0]

        data_batch.append(data_to_add)
        label_batch.extend(labels_to_add)
    return np.array(label_batch), np.array(data_batch)

#returns a one-hot label based on what stroke the datafile represents
def determine_label(stroke):
    if 'fly' in stroke.lower():
        return [1, 0, 0, 0]
    elif 'back' in stroke.lower():
        return [0, 1, 0, 0]
    elif 'breast' in stroke.lower():
        return [0, 0, 1, 0]
    elif 'free' in stroke.lower():
        return [0, 0, 0, 1]
    else:
        raise Exception("Invalid stroke! Check your spelling and make sure there are no spaces.")

def preprocess(hparams, stroke):
    pair = [[], [[],[],[]]] #pair for feed_dict
    pair[0].append(determine_label(stroke))

    for filename in glob.glob(hparams.data_dir + '*' + stroke + '*.txt'):

        print("FILENAME IS: {0}".format(filename))

        total_accs, line_count = allocate_data(filename)
        for i in range(len(pair[1])):
            pair[1][i].extend(total_accs[i])

        print("**** LENGTH OF PAIR[0] IS: {0}".format(len(pair[0])) +" ****")
        print("**** LENGTH OF PAIR[1] IS: {0}".format(len(pair[1])) +" ****")
        print("**** LENGTH OF Y ACCELERATIONS IS: {0}".format(len(pair[1][1])) +" ****")
    return np.array(pair)

def write_part(begin, end, filename):
    with open(filename, 'r') as r:
        with open(file, 'a+') as a:
            for i in range(begin):
                r.readline()
            for i in range(end):
                line = r.readline()
                a.write(line)

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
            with open("aug_" + filename, "w+") as aug:
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
    print("*******NUM_MEASUREMENTS={0}".format(num_measurements))
    data_batch, label_batch = [], []
    for i in range(hparams.batch_size):
        random_index = random.randint(0, num_measurements - hparams.batch_size)
        #this transposes the arr to allow it to fit the shape of the placeholder
        data_to_add = list(zip(pair_input[1][0][random_index:random_index + hparams.batch_size],
                               pair_input[1][1][random_index:random_index + hparams.batch_size],
                               pair_input[1][2][random_index:random_index + hparams.batch_size],))
        # data_to_add = [pair_input[1][0][random_index:random_index + hparams.batch_size]] + \
        #               [pair_input[1][1][random_index:random_index + hparams.batch_size]] + \
        #               [pair_input[1][2][random_index:random_index + hparams.batch_size]]
        labels_to_add = pair_input[0]
        # data_batch = np.append(data_batch, total_accs[1][0:len(total_accs[1]),
        #                                                  random_index:random_index + hparams.batch_size], axis=0)
        # label_batch = np.append(label_batch, total_accs[0][random_index:random_index + hparams.batch_size], axis=0)
        data_batch.append(data_to_add)
        label_batch.extend(labels_to_add)
    print(len(label_batch[0]))
    return np.array(label_batch), np.array(data_batch)

#returns a one-hot label based on what stroke the datafile represents
def determine_label(stroke):
    if stroke.lower() == 'butterfly':
        return [1, 0, 0, 0]
    elif stroke.lower() == 'backstroke':
        return [0, 1, 0, 0]
    elif stroke.lower() == 'breastroke':
        return [0, 0, 1, 0]
    elif stroke.lower() == 'freestyle':
        return [0, 0, 0, 1]
    else:
        raise Exception("Invalid stroke! Check your spelling and make sure there are not spaces.")
    # if "butterfly" in filename.lower():
    #     return [1, 0, 0, 0]
    # elif "backstroke" in filename.lower():
    #     return [0, 1, 0, 0]
    # elif "breastroke" in filename.lower():
    #     return [0, 0, 1, 0]
    # elif "freestyle" in filename.lower():
    #     return [0, 0, 0, 1]
    # else:
    #     raise Exception('a stroke name must be present in the file name')

def preprocess(hparams, stroke):
    pair = [[], []] #pair for feed_dict
    #os.chdir(hparams.data_dir)
    for filename in glob.glob(hparams.data_dir + '*' + stroke + '*.txt'):
        #arr of z, y, x accelerations over time
        #stroke_data = [[],[],[]]
        total_accs, line_count = allocate_data(filename)
        # for i in range(line_count):
        #     pair[0].append(label_vector)
        pair[1].extend(total_accs)
        pair[0].append(determine_label(stroke))
        # for line in f:
        #     accs = line.strip('\n').split(',') #puts z y x accelerations into array
        #     accs = cast_and_fix_bad_data(accs)
        #     stroke_data[0].append(accs[0])
        #     stroke_data[1].append(accs[1])
        #     stroke_data[2].append(accs[2])
        #     pair[0].extend(label_vector)
        # exclude = set(range(115, 189)) | set(range(278, 302)) | set(range(480, 505)) |\
        #             set(range(688, 752)) | set(range(903, 956)) | set(range(1104, 1137)) | set(range(1274, 1313)) | set(range(1427, 4000))
        # stroke_data[0] = [el for el in stroke_data[0] if el not in exclude]
        # stroke_data[1] = [el for el in stroke_data[1] if el not in exclude]
        # stroke_data[2] = [el for el in stroke_data[2] if el not in exclude]
        # exclude = range(128, 4000)
        # stroke_data[0] = [stroke_data[0][i] for i in range(count) if i not in exclude]
        # stroke_data[1] = [stroke_data[1][i] for i in range(count) if i not in exclude]
        # stroke_data[2] = [stroke_data[2][i] for i in range(count) if i not in exclude]
        #pair[1].extend(stroke_data)
        print("**** LENGTH OF PAIR[0] IS: {0}".format(len(pair[0])) +" ****")
        print("**** LENGTH OF Y ACCS IS: {0}".format(len(pair[1][1])) +" ****")
    return np.array(pair)

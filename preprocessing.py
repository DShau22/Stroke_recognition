import hyperparameters
import random
import os
import glob
from math import cos, sin
#creates a big ass list: each entry is a pair, first is label, second is acc data, pop it while training
#For now, just segments a sample data text file into strokes
hparams = hyperparameters.hparams

#rotates axes of measurement by rotation_angle as defined in hparams and
#stores the rotated data into aug_data folder
def augment_data(raw_data_dir, aug_data_dir):
    os.chdir(raw_data_dir)
    for filename in glob.glob('*.txt'):
        print(filename)
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
    #text file has x, y, z accs in z, y, x \n format for every .1 secs
    total_accs = np.array([[],[],[]])
    with open(text_file_path, 'r') as f:
        #take out first minute
        for i in range(600):
            f.readline()
        for line in f:
            accs = line.strip('\n').split(',')
            accs = cast_and_fix_bad_data(accs)
            #accs should now be arr of length 3 with integer entries
            total_accs[0].append(accs[0])
            total_accs[1].append(accs[1])
            total_accs[2].append(accs[2])
        return total_accs

#splits data from numpy array into batch_size number of random segments of shape (3, window size)
def make_batch(total_accs):
    num_measurements = len(total_accs[0]) #number of accerelation measurements per axis
    random_index = random.randint(0, num_measurements - hparams.batch_size - 1)
    batch = np.array([])
    for i in range(hparams.batch_size):
        batch.append(total_accs[0:len(total_accs),
                                random_index:random_index + hparams.batch_size])
    return batch

def preprocess(): #used to test if train works for now
     #arr of x, y, z accelerations over time
    stroke_data = [[],[],[]] #for now just butterfly
    pair = [[], []] #pair for feed_dict
    with open("data/Cubby's 200im right.txt", "r") as f:
        count = 0
        for line in f:
            accs = line.strip('\n').split(',') #puts x y z accelerations into array
            accs = cast_and_fix_bad_data(accs)
            stroke_data[0].append(accs[0])
            stroke_data[1].append(accs[1])
            stroke_data[2].append(accs[2])
            count += 1
        print("total number of lines: {0}".format(count))
        # exclude = set(range(115, 189)) | set(range(278, 302)) | set(range(480, 505)) |\
        #             set(range(688, 752)) | set(range(903, 956)) | set(range(1104, 1137)) | set(range(1274, 1313)) | set(range(1427, 4000))
        # stroke_data[0] = [el for el in stroke_data[0] if el not in exclude]
        # stroke_data[1] = [el for el in stroke_data[1] if el not in exclude]
        # stroke_data[2] = [el for el in stroke_data[2] if el not in exclude]
        exclude = range(128, 4000)
        stroke_data[0] = [stroke_data[0][i] for i in range(count) if i not in exclude]
        stroke_data[1] = [stroke_data[1][i] for i in range(count) if i not in exclude]
        stroke_data[2] = [stroke_data[2][i] for i in range(count) if i not in exclude]
        pair[1].extend(stroke_data)
        pair[0].extend([1, 0, 0, 0]) #one-hot encoded for butterfly
        print("**** LENGTH OF PAIR[1] IS: {0}".format(len(pair[1])) +"****")
        print("**** LENGTH OF Y ACCS IS: {0}".format(len(pair[1][1])) +"****")
    return pair

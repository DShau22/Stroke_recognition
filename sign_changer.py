import os
import preprocessing

def change_sign(filename):
    with open(filename, "r") as f:
        with open(filename + "_sign_changed.txt", "w+") as free:
            for line in f:
                accs_data = line.strip('\n').split(',')
                accs_data = preprocessing.cast_and_fix_bad_data(accs_data)
                accs_data[1], accs_data[2] = -1*accs_data[1], -1*accs_data[2]
                free.write("%d,%d,%d\n" % (accs_data[0], accs_data[1], accs_data[2]))

import os
import preprocessing

def change_sign(filename):
    with open(filename, "r") as f:
        with open(filename[:-4] + "_sign_changed.txt", "w+") as new:
            for line in f:
                accs_data = line.strip('\n').split(',')
                accs_data = preprocessing.cast_and_fix_bad_data(accs_data)
                accs_data[0], accs_data[1], accs_data[2] = -1*accs_data[0], -1*accs_data[1], -1*accs_data[2],
                new.write("%d,%d,%d\n" % (accs_data[0], accs_data[1], accs_data[2]))

# change_sign("./stroke_data/butterfly.txt")

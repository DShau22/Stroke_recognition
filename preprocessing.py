#creates a big ass list: each entry is a pair, first is label, second is acc data, pop it while training
#For now, just segments a sample data text file into strokes

def preprocess():#num_turns, exclude_ranges):
     #arr of x, y, z accelerations over time
    stroke_data = [[],[],[]] #for now just butterfly
    pair = [[], []] #pair for feed_dict
    with open("data/Cubby's 200im right.TXT", "r") as f:
        count = 0
        for line in f:
            line = line.strip('\n')
            accs = line.split(",") #puts x y z accelerations into array
            if any(["-" == el for el in accs]):
                for i in range(len(accs)):
                    try:
                        accs[accs.index("-")] = 0
                    except ValueError:
                        pass
            while len(accs) < 3:
                accs.append(0)
            stroke_data[0].append(int(accs[0]))
            stroke_data[1].append(int(accs[1]))
            stroke_data[2].append(int(accs[2]))
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

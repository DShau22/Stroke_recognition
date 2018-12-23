import graph
import train
import preprocessing
import hyperparameters
import sign_changer

def main():

    hparams = hyperparameters.hparams

    print("************************************")
    print("*********** Begin Train ************")
    print("************************************")

    print("Log directory: %s" % hparams.log_dir)
    #don't actually use loss for now
    loss, optimizer, accuracy, data_placeholder, label_placeholder = graph.build_graph(hparams)
    sign_changer.function()
    #input_data = preprocessing.preprocess()
    #preprocessing.augment_data(hparams.raw_data_dir, hparams.aug_data_dir)
    """pseudocode"""
    # augment_all_data()
    # fly = split_data(fly)
    # back = split_data(back)
    # breast = split_data(breast)
    # free = split_data(free)
    # input_data = make_batch(fly)
    #train.run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy, input_data)

if __name__ == '__main__':
	main()

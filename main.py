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
    #sign_changer.change_sign("./free_with_back.txt")
    #don't actually use loss for now
    loss, optimizer, accuracy, data_placeholder, label_placeholder, summaries = graph.build_graph(hparams)
    input_pairs = []
    #input_pairs.append(preprocessing.preprocess(hparams, 'butterfly'))
    input_pairs.append(preprocessing.preprocess(hparams, 'backstroke'))
    #input_pairs.append(preprocessing.preprocess(hparams, 'breastroke'))
    input_pairs.append(preprocessing.preprocess(hparams, 'freestyle'))

    #preprocessing.augment_data(hparams)
    train.run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy, input_pairs, summaries)

if __name__ == '__main__':
	main()

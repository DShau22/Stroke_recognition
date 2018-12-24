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
    loss, optimizer, accuracy, data_placeholder, label_placeholder, summaries = graph.build_graph(hparams)
    print(summaries)
    input_data = preprocessing.preprocess(hparams, 'freestyle')
    #preprocessing.augment_data(hparams.raw_data_dir, hparams.aug_data_dir)
    train.run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy, input_data, summaries)

if __name__ == '__main__':
	main()

import graph
import train
import hyperparameters

def main():

	hparams = hyperparameters.hparams

	print("************************************")
	print("*********** Begin Train ************")
	print("************************************")

	print("Log directory: %s" % hparams.log_dir)

	loss, optimizer, accuracy, data_placeholder, label_placeholder = graph.build_graph(hparams) #don't actually use loss for now
	train.run_training(hparams, data_placeholder, label_placeholder, optimizer, accuracy)

if __name__ == '__main__':
	main()

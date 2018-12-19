import tensorflow as tf

FS = 16000
TIME_LENGTH = 2
FRAME_LENGTH = 512

hparams = tf.contrib.training.HParams(

    #graph parameters
    num_Convs = 3,
    num_FCs = 2,
    fc_size = 256,
    window_size = 128, #change later, corresponds to Height/H
    input_width = 1,
    input_channels = 3,
    num_classes = 4, #there are 4 strokes, if not swimming then prob in classification should be low, so use threshold
    treshold_prob = .3,
    dropout_rate = .5,

    #conv1 Parameters
    filter_height1 = 7,
    stride1 = [1, 1, 1, 1],
    num_filters1 = 72,
    filter_width1 = 1,

    #conv2 parameters
    filter_height2 = 6,
    stride2 = [1, 1, 1, 1],
    num_filters2 = 144,
    filter_width2 = 1,

    #conv3 parameters
    filter_height3 = 5,
    stride3 = [1, 1, 1, 1],
    num_filters3 = 108,
    filter_width3 = 1,

    #Training parameters
    max_iterations = 10, #int(1e6),
    print_loss_freq = 32,
    writing_frequency = 32,
    learning_rate = .0003,
	max_steps = 10,
	batch_size = 1,
	train_loss_frequency = 16,
	val_loss_frequency = 32,

	# Logging Parameters
	log_dir = "logs/",
	log_file_train = "logs/loss.txt",
	log_file_val = "logs/loss_val.txt",

	# Data Parameters
	raw_data_dir = "./data/",
    aug_data_dir = "./aug_data/",
    rotation_angle = .2, #in radians

	print_loss_frequency = 16,
	save_model_interval = 600,
	save_dir = "checkpoints/",

	)

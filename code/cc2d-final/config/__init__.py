from yacs.config import CfgNode as CN
import os

__C = CN()

__C.dataset = CN()
__C.dataset.dataset_pth = ''
__C.dataset.num_landmarks = 19
__C.dataset.eval_radius = []

__C.dataset.AUGMENTATION = CN(new_allowed=True)
__C.dataset.AUGMENTATION.REVERSE_AXIS = False
__C.dataset.AUGMENTATION.FLIP = False
__C.dataset.AUGMENTATION.FLIP_PAIRS = []
__C.dataset.AUGMENTATION.ROTATION_FACTOR = 3
__C.dataset.AUGMENTATION.INTENSITY_FACTOR = 0.5
__C.dataset.AUGMENTATION.SF = 0.05
__C.dataset.AUGMENTATION.TRANSLATION_X = 10
__C.dataset.AUGMENTATION.TRANSLATION_Y = 10
__C.dataset.AUGMENTATION.ELASTIC_STRENGTH = 500
__C.dataset.AUGMENTATION.ELASTIC_SMOOTHNESS = 30

__C.train = CN()
__C.train.learning_rate = 0.001
__C.train.batch_size = 8
__C.train.decay_step = 50
__C.train.decay_gamma = 0.5
__C.train.num_epochs = 300
__C.train.num_workers = 8
__C.train.save_seq = 10
__C.train.loss_lambda = 0.001
__C.train.input_size = []
__C.test = CN()
__C.test.id_oneshot = []

__C.args = CN()

def get_cfg_defaults():
	return __C.clone()

def merge_cfg_datasets(cfg, dataset='head'):
	assert dataset in ['head', 'hand', 'leg', 'chest']
	current_file_path = os.path.abspath(__file__)
	current_directory = os.path.dirname(current_file_path)
	dataset_yaml = os.path.join(current_directory, f'{dataset}.yaml')
	cfg.merge_from_file(dataset_yaml)
	cfg.dataset.dataset = dataset

def merge_cfg_train(cfg, train='voting'):
	assert hasattr(cfg.dataset, 'dataset')
	assert train in ['SSL', 'voting', 'heatmap', 'TPL', 'SAM', 'ERE']
	current_file_path = os.path.abspath(__file__)
	current_directory = os.path.dirname(current_file_path)
	dataset_yaml = os.path.join(current_directory, f'{train}_{cfg.dataset.dataset}.yaml')
	cfg.merge_from_file(dataset_yaml)

def merge_from_args(cfg, args, key_list=None):
	args_dict = vars(args)
	if key_list is None:
		for key, value in args_dict.items():
			cfg.args[key] = value
	else:
		for key in key_list:
			cfg.args[key] = args_dict[key]
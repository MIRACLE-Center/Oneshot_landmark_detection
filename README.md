# oneshot-medical-landmark
Implementation of "One-shot Medical Landmark Detection" -- MICCAI 2021

Paper link: https://arxiv.org/abs/2103.04527

## Usage

#### Environment

python >= 3.5; pytorch >= 1.1.0; torchvision >= 0.6

extra package: **tutils** 

`pip install git+http://gitee.com/transcendentsky/tutils.git`

 

#### Data Preparation

We train/test our model on Dataset: Cephalometric

We expect the dictionary structure to be the following:

````
path/to/cephalometric
	400_junior
		001.txt
		...
	400_senior
		001.txt
		...
	RawImage
		TrainingData
			001.bmp
			...
		Test1Data
			151.bmp
			...
		Test2Data
			301.bmp
			...
````

## Training

Stage 1: self-supervised training

`python -m scripts.train --tag xxtag`

Stage 2: self-training
`python -m scripts.self_train --tag xxtag`

## Citation
Please ite our paper if it helps.
````
@article{yao2021one,
  title={One-Shot Medical Landmark Detection},
  author={Yao, Qingsong and Quan, Quan and Xiao, Li and Zhou, S Kevin},
  journal={arXiv preprint arXiv:2103.04527},
  year={2021}
}
````
## License
This code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information

We actively welcome your pull requests! 

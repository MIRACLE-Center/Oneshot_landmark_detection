# oneshot-medical-landmark
Implementation of "One-shot Medical Landmark Detection" -- MICCAI 2021

Paper link: https://doi.org/10.1007/978-3-030-87196-3_17

## Usage

#### Environment

python >= 3.9
```
pip install -r req.txt
```



## Data Preparation

We up load the four X-ray datasets in google drive, including our proposed BMPLE dataset (Leg X-ray). Please download the files and unzip into the diretory ``datasets''

[head dataset (Cephalometric)](https://drive.google.com/file/d/1PQhNrwZpa9y6AhWjX_qcS_zrptcTCSvq/view?usp=sharing)

[hand dataset](https://drive.google.com/file/d/16MD617NySbf7EXlkA07RlMnEUAUYgHCt/view?usp=sharing)

[chest dataset](https://drive.google.com/file/d/1Gkfl5wnaU2J-TMAkOrKM604cmxw9PPiz/view?usp=sharing)

[leg dataset (BMPLE)](https://drive.google.com/file/d/1MWCWA64xJ9Rt4MpcqinQMkJ2Fs7ekGRr/view?usp=sharing)


## Training

Stage 1: self-supervised training
```
python -m scripts.train --tag xxtag --dataset head
```

### Automatically choose the template
```
python -m scripts.SCP --tag xxtag --dataset head
```
You can set the selected id by SCP and set to config/SSL_{dataset}.yaml and config/TPL_{dataset}.yaml, otherwise you can choose to use the best template.

```
python -m scripts.train --tag xxtag --dataset head --finetune 1
```

Stage 2: self-training
```
python -m scripts.self_train --tag xxtag
```

## Citation
Please ite our paper if it helps.
````
@article{yao2021one,
  title={One-Shot Medical Landmark Detection},
  author={Yao, Qingsong and Quan, Quan and Xiao, Li and Zhou, S Kevin},
  journal={Medical Image Computing and Computer Assisted Intervention},
  year={2021}
}
````
## License
This code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information

We actively welcome your issues! 

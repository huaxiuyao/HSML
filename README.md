# HSML (Hierarchically Structured Meta-learning)

## About
Source code<a href="#note1" id="note1ref"><sup>1</sup></a> of the paper [Hierarchically Structured Meta-learning](https://arxiv.org/abs/1905.05301)

For continual version of this algorithm, please refer to this <a href="#note1" id="note1ref"></a> [repo](https://github.com/huaxiuyao/HSML_Dynamic).

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2019hierarchically,
  title={Hierarchically Structured Meta-learning},
  author={Yao, Huaxiu and Wei, Ying and Huang, Junzhou and Li, Zhenhui},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019} 
}
```

## Data
We release our Multi-Datasets including bird, texture, aircraft and fungi in this [link](https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view?usp=sharing).

## Usage

### Dependence
* python 3.*
* TensorFlow 1.0+
* Numpy 1.15+

### Toy Group Data
Please see the bash file in /toygroup_bash for parameter settings

### Multi-datasets Data
Please see the bash file in /multidataset_bash for parameter settings


<a id="note1" href="#note1ref"><sup>1</sup></a>This code is built based on the [MAML](https://github.com/cbfinn/maml).


# Towards Generic Image Manipulation Detection with Weakly-Supervised Self-Consistency Learning


<p align="center">
  <a href="https://arxiv.org/abs/2309.01246"><b>Towards Generic Image Manipulation Detection with Weakly-Supervised Self-Consistency Learning</b></a>
</p>

<p align="center">
  <a href="https://www.yhzhai.com">Yuanhao Zhai</a>, 
  <a href="https://tyluann.github.io">Tianyu Luan</a>, 
  <a href="https://cse.buffalo.edu/~doermann/">David Doermann</a>, 
  <a href="https://cse.buffalo.edu/~jsyuan/">Junsong Yuan</a>
</p>

<p align="center">
  University at Buffalo
</p>

<p align="center">
  ICCV 2023
</p>

![tp](./assets/tp.jpg)

This repo contains the MIL-FCN version of our WSCL implementation.

## 🚨News

**03/2024**: add demo script! Check [here](https://github.com/yhZhai/WSCL?tab=readme-ov-file#4-demo) for more details!

## 1. Setup
Clone this repo

```bash
git clone git@github.com:yhZhai/WSCL.git
```

Install packages
```bash
pip install -r requirements.txt
```

## 2. Data preparation

We provide preprocessed CASIA (v1 and v2), Columbia, and Coverage datasets [here](https://buffalo.box.com/s/2t3eqvwp7ua2ircpdx12sfq04sne4x50).
Place them under the `data` folder.

For other datasets, please prepare a json datalist file with similar structure as the existing datalist files in the `data` folder. After that, adjust the `train_dataslist` or the `val_datalist` entries in the configuration files `configs/final.yaml`.


## 3. Training and evaluation

Runing the following script to train on CASIAv2, and evalute on CASIAv1, Columbia and Coverage.

```shell
python main.py --load configs/final.yaml
```

For evaluating a pre-trained checkpoint:
```shell
python main.py --load configs/final.yaml --eval --resume checkpoint-path
```

We provide our pre-trained checkpoint [here](https://buffalo.box.com/s/2t3eqvwp7ua2ircpdx12sfq04sne4x50).


## 4. Demo

Running our manipulation model on your custom data!
Before running, please configure your desired input and output path in the `demo.py` file.

```shell
python demo.py --load configs/final.yaml --resume checkpoint-path
```

By default, it evaluates all  `.jpg` files in the `demo` folder, and saves the
detection result in `tmp`, with manipulation probablities appended to the file names.



## Citation
If you feel this project is helpful, please consider citing our paper
```bibtex
@inproceedings{zhai2023towards,
  title={Towards Generic Image Manipulation Detection with Weakly-Supervised Self-Consistency Learning},
  author={Zhai, Yuanhao and Luan, Tianyu and Doermann, David and Yuan, Junsong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22390--22400},
  year={2023}
}
```


## Acknowledgement
We would like to thank the following repos for their great work:
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [DETR](https://github.com/facebookresearch/detr)

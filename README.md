# Towards Generic Image Manipulation Detection with Weakly-Supervised Self-Consistency Learning


> [**Towards Generic Image Manipulation Detection with Weakly-Supervised Self-Consistency Learning**](https://arxiv.org/abs/2309.01246)
>
> [Yuanhao Zhai](https://www.yhzhai.com), [Tianyu Luan](https://tyluann.github.io), [David Doermann](https://cse.buffalo.edu/~doermann/), [Junsong Yuan](https://cse.buffalo.edu/~jsyuan/)
>
> University at Buffalo
>
> ICCV 2023

This repo contains the MIL-FCN version of our WSCL implementation.

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


## 3. Training and evaluation

Runing the following script to train on CASIAv2, and evalute on CASIAv1, Columbia and Coverage.

```shell
python main.py --load configs/final.yaml
```


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

# SDS-CL
Spatiotemporal Decouple-and-Squeeze Contrastive Learning for Semisupervised Skeleton-Based Action Recognition
## Requirements
- python == 3.8.3
- pytorch == 1.11.0
- CUDA == 11.2
## Data Preparation
Download the raw data of [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D), [NTU-RGB+D 120](https://github.com/shahroudy/NTURGB-D), [NW-UCLA](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0), and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn).
## Training
```
python train_val_test/train.py -config ./train_val_test/config/ntu/ntu60_dstanet.yaml
python train_val_test/train_finetune.py -config ./train_val_test/config/ntu/ntu60_dstanet_finetune.yaml
```
## Acknowledgements
This repo is based on [DSTA-Net](https://github.com/lshiwjx/DSTA-Net), thanks to the original authors for their works!
## Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{xu2023spatiotemporal,
  title={Spatiotemporal Decouple-and-Squeeze Contrastive Learning for Semisupervised Skeleton-Based Action Recognition},
  author={Xu, Binqian and Shu, Xiangbo and Zhang, Jiachao and Dai, Guangzhao and Song, Yan},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}
 ```

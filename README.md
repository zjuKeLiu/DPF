# Boost Your Crystal Model with Denoising Pre-training


[[OpenReview](https://openreview.net/forum?id=u2qYzRRg02)] [[Project Page](https://ai4mol.github.io/projects/DPF)]

## News

Stay updated with the latest milestones of our work:

- 🌟 **Featured at [ICML AI4Science Workshop 2024](https://openreview.net/forum?id=u2qYzRRg02)**  
  *"Boost Your Crystal Model with Denoising Pre-training"*

- 🌟 **Presented at [AAAI Conference 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35058)**  
  *"A Denoising Pre-training Framework for Accelerating Material Discovery"*

## Dataset

The dataset used for pre-training can be found at [GNoME](https://github.com/google-deepmind/materials_discovery).

## Training and Prediction

You can train and test the model with the following commands:

```bash
conda env create -f DPF.yaml
conda activate DPF
cd matformer
bash pretrain.sh
```

For training your own custom models, you only need to replace the model with your own.

## Citation
Please cite our paper if you find the code helpful.
```
@inproceedings{DPF_ICML,
  title={Boost Your Crystal Model with Denoising Pre-training},
  author={Shuaike Shen and Ke Liu and Muzhi Zhu and Hao Chen},
  booktitle={ICML 2024 AI for Science Workshop},
  year={2024},
  url={https://openreview.net/forum?id=u2qYzRRg02}
}

@inproceedings{DPF_AAAI,
  title={A Denoising Pre-training Framework for Accelerating Novel Material Discovery},
  author={Shen, Shuaike and Liu, Ke and Zhu, Muzhi and Chen, Hao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28368--28376},
  year={2025}
}

@inproceedings{ijcai2022p708,
  title     = {S2SNet: A Pretrained Neural Network for Superconductivity Discovery},
  author    = {Liu, Ke and Yang, Kaifan and Zhang, Jiahong and Xu, Renjun},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {5101--5107},
  year      = {2022},
  month     = {7},
  doi       = {10.24963/ijcai.2022/708},
  url       = {https://doi.org/10.24963/ijcai.2022/708},
}
```

## Acknowledgement

This repo is built upon the previous work ALIGNN and MatFormer. The original idea comes from S2SNet.

## Contact

If you have any questions, please contact me at kliu@zju.edu.cn

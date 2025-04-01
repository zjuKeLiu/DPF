# Boost Your Crystal Model with Denoising Pre-training


[[OpenReview](https://openreview.net/forum?id=u2qYzRRg02)] [[Project Page](https://ai4mol.github.io/projects/DPF)]

## News

Stay updated with the latest milestones of our work:

- 🌟 **Featured at [ICML AI4Science Workshop 2024](https://openreview.net/forum?id=u2qYzRRg02)**  
  *"Boost Your Crystal Model with Denoising Pre-training"*

- 🌟 **Presented at [AAAI Conference 2025 （to appear）](https://aaai.org/)**  
  *"A Denoising Pre-training Framework for Accelerating Material Discovery"*

## Dataset

Dataset used for pre-training can be found at [GNoME](https://github.com/google-deepmind/materials_discovery).

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
@inproceedings{shen2024boost,
  title={Boost Your Crystal Model with Denoising Pre-training},
  author={Shuaike Shen and Ke Liu and Muzhi Zhu and Hao Chen},
  booktitle={ICML 2024 AI for Science Workshop},
  year={2024},
  url={https://openreview.net/forum?id=u2qYzRRg02}
}
```

## Acknowledgement

This repo is built upon the previous work ALIGNN and MatFormer

## Contact

If you have any question, please contact me at kliu@zju.edu.cn

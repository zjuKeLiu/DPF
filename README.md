# Boost Your Crystal Model with Denoising Pre-training


[[OpenReview](https://openreview.net/forum?id=u2qYzRRg02)] [[Project Page](https://ai4mol.github.io/projects/DPF)]

The official implementation of Boost Your Crystal Model with Denoising Pre-training (ICML AI4Science Workshop 2024) & A Denoising Pre-training Framework for Accelerating Novel Material Discovery (AAAI 2025).



## Training and Prediction

You can train and test the model with the following commands:

```bash
conda env create -f DPF.yaml
conda activate DPF
cd matformer
bash pretrain.sh
```

## Citation
Please cite our paper if you find the code helpful or if you want to use the benchmark results of the Materials Project and JARVIS. Thank you!
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

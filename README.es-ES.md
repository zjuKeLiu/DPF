   [Done] 
   - *Output Generation* (matches the final refined version) -> *Proceeds*
</think>

# Mejora tu modelo de cristales con preentrenamiento por denoising

[[OpenReview](https://openreview.net/forum?id=u2qYzRRg02)] [[Project Page](https://ai4mol.github.io/projects/DPF)]

## Novedades

Mantente al día con los últimos hitos de nuestro trabajo:

- 🌟 **Destacado en [ICML AI4Science Workshop 2024](https://openreview.net/forum?id=u2qYzRRg02)**  
  *"Mejora tu modelo de cristales con preentrenamiento por denoising"*

- 🌟 **Presentado en [AAAI Conference 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35058)**  
  *"Un marco de preentrenamiento por denoising para acelerar el descubrimiento de materiales"*

## Conjunto de datos

El conjunto de datos utilizado para el preentrenamiento se puede encontrar en [GNoME](https://github.com/google-deepmind/materials_discovery).

## Entrenamiento y Predicción

Puedes entrenar y probar el modelo con los siguientes comandos:

```bash
conda env create -f DPF.yaml
conda activate DPF
cd matformer
bash pretrain.sh
```

Para entrenar tus propios modelos personalizados, solo necesitas reemplazar el modelo por el tuyo.

## Cita

Por favor, cita nuestro artículo si consideras que el código es útil.
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

## Agradecimientos

Este repositorio se basa en los trabajos previos ALIGNN y MatFormer. La idea original proviene de S2SNet.

## Contacto

Si tienes alguna pregunta, por favor contáctame a kliu@zju.edu.cn

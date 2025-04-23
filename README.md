# BCE
This is an official PyTorch implementation of the paper "Enhancing Clinical Information for Zero-Shot Medical Diagnosis by Prompting Large Language Model"

## Introduction

In real clinical diagnosis workflow, unseen disease categories are commonly encountered, where most existing supervised deep learning methods are invalid to accurately recognize.  Recent works utilizing large-scale image-report datasets to train vision-language models have witnessed impressive zero-shot capabilities, while they rely on high-quality diagnosis reports that are difficult to collect, especially on some rare diseases. In this work, we propose Bidirectional vision-language Clinical information Exploitation (BCE), a new paradigm towards superior generalized zero-shot learning for medical diagnosis by multi-modal information mining. To harvest sparse disease semantics in medical images, the Cross-modal Knowledge Interaction (CKI) is designed by matching the global textual information towards local visual representations, which encourages the model to capture dense correspondence from visual to textual information. Furthermore, instead of using category keywords as text prompts to yield fixed descriptions from large language models (LLM) in previous works, we propose a Modality-Guided model Tuning (MGT) to encourage the LLM to produce fine-grained clinical information conditioned on input visual information. MGT can efficiently update additional learnable parameters inserted into the LLM and dynamically adapt them to yield instance-aware clinical information. Finally, a Fine-grained text-image Alignment (FA) is present to provide reliable constraint for superior discrimination. Extensive experiments on various medical generalized zero-shot learning benchmarks demonstrate the superiority of the proposed framework.

## Dependencies
Dependency packages can be installed using following command:
```
pip install -r requirements.txt
```

## Quickstart

### Training
```python
sh /BCE/finetuning.sh
```

## Citation
```python
@inproceedings{yang2024enhancing,
  title={Enhancing Clinical Information for Zero-Shot Medical Diagnosis by Prompting Large Language Model},
  author={Yang, Qiushi and Zhu, Meilu and Yuan, Yixuan},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={2760--2765},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgement
* The implementation of baseline method is adapted from: [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

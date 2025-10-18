# AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition

[![arXiv](https://img.shields.io/badge/arXiv-2507.19840-b31b1b.svg)](https://arxiv.org/abs/2507.19840)
[![Competition](https://img.shields.io/badge/ICCV-MSLR%202025-blue)](https://iccv-mslr-2025.github.io/MSLR/)

A PyTorch implementation of AutoSign, a novel approach for continuous sign language recognition that directly translates pose sequences to text without intermediate gloss annotations.

> **📄 Paper:** [AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition](https://arxiv.org/abs/2507.19840)

## Demo

<div align="center">

### Signer-Independent Recognition Examples

<table>
  <tr>
    <td align="center">
      <img src="pose_animation\14_0004_pose_animation.gif" width="450px" />
      <br>
      <table>
        <tr><td><b>Ground Truth:</b></td><td>هو معمل لا إنا مدرسة</td></tr>
        <tr><td></td><td><i>HE TEACHER NO I SCHOOL</i></td></tr>
        <tr><td><b>AutoSign:</b></td><td>هو معمل لا إنا مدرسة</td></tr>
        <tr><td></td><td><i>HE TEACHER NO I SCHOOL</i></td></tr>
      </table>
    </td>
    <td align="center">
      <img src="pose_animation\14_0001_pose_animation.gif" width="450px" />
      <br>
      <table>
        <tr><td><b>Ground Truth:</b></td><td>سؤال هو</td></tr>
        <tr><td></td><td><i>QUESTION HE</i></td></tr>
        <tr><td><b>AutoSign:</b></td><td>سؤال هو</td></tr>
        <tr><td></td><td><i>QUESTION HE</i></td></tr>
      </table>
    </td>
  </tr>
</table>

</div>

## Overview

AutoSign addresses the challenge of continuous sign language recognition by leveraging pose estimation and transformer-based language models. Our approach:

- **Direct pose-to-text translation** without requiring gloss-level annotations
- **Arabic language support** using fine-tuned AraGPT2
- **State-of-the-art performance** achieving **6.7% WER** on the dev set and **20.5% WER** on the test set
- **Efficient architecture** using only pose sequences instead of RGB frames

## Dataset

This implementation is designed for the [ICCV 2025 MSLR Challenge](https://iccv-mslr-2025.github.io/MSLR/).

**Download the dataset:**
- Competition page: [Kaggle - Continuous Sign Language Recognition ICCV 2025](https://www.kaggle.com/competitions/continuous-sign-language-recognition-iccv-2025)
- Original benchmark: [Pose86K-CSLR-Isharah Repository](https://github.com/gufranSabri/Pose86K-CSLR-Isharah)

## Installation

```bash
git clone https://github.com/yourusername/AutoSign
cd AutoSign
pip install -r requirements.txt
```

## Pre-trained Models

### Arabic GPT-2 Model

We use a fine-tuned Arabic GPT-2 model based on [AraGPT2 by AUB-MIND](https://github.com/aub-mind/arabert).

**Download the pre-trained Arabic GPT-2:**

```bash
#!/bin/bash

if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing..."
    pip install gdown
fi

mkdir -p models/gpt2

gdown --output gpt2_model.zip 1rugTrGKChYXcsjLiIrfbBCBj4LOwVsvy

unzip -q gpt2_model.zip -d models/gpt2

rm gpt2_model.zip

echo "Setup complete. GPT2 model is in models/gpt2/"
```

### AutoSign Checkpoints

Pre-trained AutoSign checkpoints are available upon request for research purposes.

**To access the checkpoints:**
- Email: [sjohnny@andrew.cmu.edu](mailto:sjohnny@andrew.cmu.edu)
- Please include your name, affiliation, and intended use case

## Training

To train the AutoSign model from scratch:

```bash
python main.py
```

## Evaluation

### Development Set Evaluation

```bash
python dev_script.py
```

The results will be automatically saved in `.zip` format.

### Test Set Evaluation

```bash
python test_script.py
```

## Results

### Word Error Rate (WER) Comparison

| Method | Input | WER (%) Dev | WER (%) Test |
|--------|-------|-------------|--------------|
| VAC [33] | RGB | 18.9 | 31.9 |
| SMKD [15] | RGB | 18.5 | 35.1 |
| TLP [19] | RGB | 19.0 | 32.0 |
| SEN [22] | RGB | 19.1 | 36.4 |
| CorrNet [21] | RGB | 18.8 | 31.9 |
| Swin-MSTP [5] | RGB | 17.9 | 26.6 |
| SlowFastSign [1] | RGB | 19.0 | 32.1 |
| Baseline | Pose | 20.5 | 33.2 |
| **AutoSign (Ours)** | **Pose** | **6.7** | **20.5** |

**AutoSign achieves state-of-the-art performance** with a 67% relative improvement over the baseline on the Dev set and 38% improvement on the Test set, while using only pose data instead of RGB frames.

## Architecture

AutoSign employs a decoder-only transformer architecture that:

1. **Pose Encoding**: Processes skeletal pose sequences from video frames
2. **Language Modeling**: Leverages pre-trained Arabic GPT-2 for sequence generation
3. **Direct Translation**: Maps pose features directly to text without gloss intermediates

## Citation

If you use AutoSign in your research, please cite:

```bibtex
@article{johnny2025autosign,
  title={AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition},
  author={Johnny, Samuel Ebimobowei and Guda, Blessed and Stephen, Andrew Blayama and Gueye, Assane},
  journal={arXiv preprint arXiv:2507.19840},
  year={2025}
}
```

## Acknowledgments

This project is inspired by and builds upon several foundational works:

- **DTrOCR**: We acknowledge the decoder-only transformer architecture principles from [DTrOCR: Decoder-only Transformer for Optical Character Recognition](https://doi.org/10.48550/arXiv.2308.15996) by Masato Fujitake, which informed our architectural design.

- **AraGPT2**: Our implementation leverages the Arabic GPT-2 model developed by [AUB-MIND](https://github.com/aub-mind/arabert), which provides robust Arabic language understanding capabilities.

- **Hugging Face**: We extend our gratitude to the Hugging Face team for their Transformers library, particularly the GPT-2 and Vision Transformer (ViT) implementations that form the backbone of our system.

- **ICCV MSLR 2025**: Thanks to the organizers of the [ICCV 2025 Multilingual Sign Language Recognition Challenge](https://iccv-mslr-2025.github.io/MSLR/) for providing the benchmark dataset and evaluation framework.

<!-- ## License

[Add your license here - MIT, Apache 2.0, etc.] -->

## Contributors

- Samuel Ebimobowei Johnny
- Blessed Guda
- Andrew Blayama Stephen
- Assane Gueye

**Upanzi Lab, Carnegie Mellon University Africa**

## Contact

For questions, issues, or collaboration opportunities:
- Email: [sjohnny@andrew.cmu.edu](mailto:sjohnny@andrew.cmu.edu)
- Issues: [GitHub Issues](https://github.com/yourusername/AutoSign/issues)

---

<div align="center">
Made with care for the Sign Language Recognition Community
</div>
## NYCU Selected Topics in Visual Recognition using Deep Learning HW4
- Student ID: 111550172
- Name: 游承曦

### Introduction
This assignment focuses on image restoration under degradations caused by rain and snow. Traditional deep learning-based restoration models often require separate networks for each degradation type and assume prior knowledge about the specific corruption. We adopt PromptIR, an all-in-one blind image restoration framework that dynamically adapts to multiple unknown degradation types using a single unified model.

The key design of PromptIR lies in its use of prompts—a set of tunable parameters that encode degradation-specific context. These prompts are generated through a Prompt Generation Module (PGM) and interact with feature representations via a Prompt Interaction Module (PIM), enriching them with contextual knowledge at multiple stages of the decoder.

This prompt-based design enables PromptIR to generalize across diverse degradation types without relying on task-specific branches. As a result, it serves as an effective baseline for our image restoration task. To achieve better performance, we enhance the training process with data augmentation strategies and incorporate various loss functions.

This code is based on the [PromptIR: Prompting for All-in-One Blind Image Restoration (NeurIPS'23)](https://github.com/va1shn9v/PromptIR) repositories.

### Installation
```bash
git clone https://github.com/thomasyu9393/NYCU-VRDL.git
cd NYCU-VRDL/HW4
conda create -n hw_env python=3.9
conda activate hw_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Training
```bash
python train.py --perceptual --ssim
```

### Testing
```bash
python test.py --ckpt ./epoch_1.pth --output_dir tmp
```

### Performance Snapshot
<p align="center">
  <img src="./figures/0.png" width="80%">
</p>

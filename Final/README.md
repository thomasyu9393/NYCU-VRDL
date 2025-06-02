## NYCU Selected Topics in Visual Recognition using Deep Learning Final Project
- Group: 17
- Student ID: 111550034, 111550084, 111550150, 111550172
- Name: 黃皓君, 林辰恩, 俞祺譯, 游承曦

### Introduction
In this project, our goal is to detect and count five classes of Steller sea lions—adult males, subadult males, adult females, juveniles, and pups—from high-resolution aerial images provided by the [NOAA Fisheries Steller Sea Lion Population Count Kaggle challenge](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count/overview). Our input consists of plane-captured photographs, and our output is a per-image tally for each category. We adopt the second-place solution's two-stage approach, which combines a U-Net–based density prediction with an ensemble of regression models for count estimation. Special thanks to Konstantin Lopuhin for sharing this [elegant methodology](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count/discussion/35422) and its [code](https://github.com/lopuhin/kaggle-lions-2017)!

### Installation
```bash
git clone https://github.com/thomasyu9393/NYCU-VRDL.git
cd NYCU-VRDL/Final
conda create -n hw_env python=3.9
conda activate hw_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
mkdir -p data _runs
cd data
7z x KaggleNOAASeaLions.7z -p{Password} -bb0
```

### Training UNet (~5hr)
```bash
python unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 --stratified --batch-size 32 --min-scale 0.8 --max-scale 1.6 --n-epochs 10 --oversample 0.2
ls -l _runs/unet-stratified-scale-0.8-1.6-oversample0.2/best-model.pt
```

### Testing UNet (~3hr)
```bash
python unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 --stratified --batch-size 32 --min-scale 0.8 --max-scale 1.6 --n-epochs 10 --oversample 0.2 --mode predict_all_valid
ls -l _runs/unet-stratified-scale-0.8-1.6-oversample0.2/ | grep '\-pred.npy' | head

python unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 --stratified --batch-size 32 --min-scale 0.8 --max-scale 1.6 --n-epochs 10 --oversample 0.2 --mode predict_test
ls -l _runs/unet-stratified-scale-0.8-1.6-oversample0.2/test/ | grep '\-pred.npy' | head
```

### Training Regressors (~5min)
```bash
python make_submission.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 train --explain
ls -l _runs/unet-stratified-scale-0.8-1.6-oversample0.2/regressor.joblib
```

### Testing Regressors (~1.5hr)
```bash
python make_submission.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 predict
ls -l _runs/unet-stratified-scale-0.8-1.6-oversample0.2/unet-stratified-scale-0.8-1.6-oversample0.2.csv
```

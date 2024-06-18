# NeWRF
NeWRF: A Deep Learning Framework for Wireless Radiation Field Reconstruction and Channel Prediction

## Usage
- simulator/ contains the simulation code. Run main.m in MATLAB to generate datasets
- train.py contains the training code. Usage:
```
python train.py --env [environment]
```
- eval.py contains the evaluation code. Usage:
```
python eval.py --env [environment] --ckpt [checkpoint file] 
```
- demo.ipynb provides visualization during the training process 
![demo outcome](Figures\output.png)
# SDCNN
This repository contains the PyTorch implementation of the paper: [**SDCNN: Self-Supervised Disentangled Convolutional Neural Network for Low-Dose CT Denoising**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10758838) 
***************************************************************************

> **Abstract:** Low-dose computed tomography (LDCT) reduces radiation exposure but suffers from high noise, impacting image quality and diagnostic accuracy. Supervised learning has helped address this challenge but requires numerous paired datasets of LDCT and normal-dose CT (NDCT) images, which limits their clinical practice. This paper proposes a novel self-supervised disentangled convolutional neural network (SDCNN) that can directly reconstruct high-quality CT images from LDCT data without the need for a clean reference. Unlike other methods that treat noise as a uniform entity, SDCNN disentangles LDCT images into noise-free images, signal-dependent noise, and signal-independent noise, aligning with the intrinsic principles of low-dose noise generation. To enhance the purity of disentanglement, we introduce the concept of combination and re-disentanglement to establish a training framework based on SDCNN. Additionally, we design self-supervised loss functions, including novel anisotropic total variation (TV) and distance loss functions, to improve the efficiency of the denoising process. The signal-guided attention (SGA) module effectively captures the relationship between signal-dependent noise and the signal across both spatial and channel dimensions. Experiments on clinical and animal data demonstrate that the proposed method performed better than all competing state-of-the-art self-supervised algorithms in noise and artifact removal. For example, compared to self-supervised algorithms, SDCNN can improve MSSIM, PSNR, and FSIM by at least 2.26%, 1.20dB, 1.23%, and GMSD is reduced by at least 1.11% on Mayo clinical data. 
***************************************************************************
### Illustration
<div align=center>
<img src="https://github.com/YuhangLiu98/SDCNN/blob/main/img/SDCNN.png" width="800"/> 
</div>

-------

### DATASET

1.The Mayo Clinic Low Dose CT by Mayo Clinic   
(I can't share this data, you should ask at the URL below if you want)  
https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/

2.The Piglet Low Dose CT by X. Yi and P. Babyn, "Sharpness-aware low-dose CT denoising using conditional generative adversarial network"
(I can't share this data, you should ask at the URL below if you want)  
https://github.com/xinario/SAGAN?tab=readme-ov-file


-------
## Installation

SDCNN can be installed from source,
```shell
git clone https://github.com/YuhangLiu98/SDCNN.git
cd SDCNN/src
```
Then, [Pytorch](https://pytorch.org/) is required, for example,
```shell script
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Lastly, other pakages are required,
```shell script
pip install -r requirements.txt
```

-------

## Use

1. run `python prep.py` to convert 'dicom file' to 'numpy array'
2. run `python train.py` to training.
3. run `python test.py` to testing.
-------

### RESULT  
<div align=center>
<img src="https://github.com/YuhangLiu98/SDCNN/blob/main/img/result1.png" width="800"/>   
<img src="https://github.com/YuhangLiu98/SDCNN/blob/main/img/result2.png" width="800"/>   
<img src="https://github.com/YuhangLiu98/SDCNN/blob/main/img/result4.png" width="800"/>   
</div>

### Citation
```shell
@ARTICLE{10758838,
  author={Liu, Yuhang and Shu, Huazhong and Chi, Qiang and Zhang, Yue and Liu, Zidong and Wu, Fuzhi and Coatrieux, Jean-Louis and Liu, Yi and Wang, Lei and Zhang, Pengcheng and Gui, Zhiguo},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={SDCNN: Self-Supervised Disentangled Convolutional Neural Network for Low-Dose CT Denoising}, 
  year={2024},
  doi={10.1109/TIM.2024.3502758}}
```
### Acknowlegements
Code borrows heavily from [CVF-SID](https://github.com/Reyhanehne/CVF-SID_PyTorch)

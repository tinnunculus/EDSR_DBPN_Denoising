# EDSR
* 이것은 홍익대학교 졸업을 위한 프로젝트입니다.
* EDSR(Enhanced Deep Residual Networks for Single Image Super-Resolution) with keras

![image](https://user-images.githubusercontent.com/36150943/68541243-57463d00-03e0-11ea-81d7-29c0299ac610.png)



## Data preprocess 
1. Dataset으로는 DIV2K를 사용하였다.(당신은 Dataset이라는 디렉토리에 DIV2k dataset을 저장해야만 한다.)
2. Test image 800, validation image 100
3. Opencv를 이용하여 모든 이미지를 50 x 50으로 분리하여 numpy 파일로 저장하였다.(proceseed이라는 디렉토리를 만들어 저장해야만한다.)
4. 총 49110개의 test patch image와 6250개의 validation patch image가 나온다.

## Train
데이터의 전처리가 완료되었다면, EDSR_MODEL 클래스의 train 메소드를 사용하여 학습하여주시길바랍니다.

```python
from edsr_model import EDSR_MODEL
ed = EDSR_MODEL()
ed.train()
```

## Predict
학습이 완료되었다면, test_image 파일에 처리하고 싶은 이미지를 저장하고 EDSR_MODEL 클래스의 pred 메소드를 사용하여 SR처리해주시길바랍니다.

```python
from edsr_model import EDSR_MODEL
ed = EDSR_MODEL(pretrain = True)
ed.pred("image_name")
```

## Dependencies
* python 3.6
* keras
* numpy
* matplot
* opencv
* skimage

## Reference
1. B Lim et al. Enhanced Deep Residual Networks for Single Image Super-Resolution. CVPR 2018
2. Keras Subpixel (Pixel-Shuffle layer) from: [Keras-Subpixel] (https://github.com/atriumlts/subpixel/blob/master/keras_subpixel.py)

# EDSR
* This page is projects for graduation from Hongik University in 2019. 
* EDSR(Enhanced Deep Residual Networks for Single Image Super-Resolution) with keras

![image](https://user-images.githubusercontent.com/36150943/68541243-57463d00-03e0-11ea-81d7-29c0299ac610.png)



## Data preprocess 
1. The Datasets are DIV2K(you have to store the datasets in the "dataset" directory)
2. Test images 800, validation images 100
3. By using opencv libraries, the DIV2k Images are separated from 2k to patch(50x50), and the patchs are stored in the "processed" directory.
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

## Result

![image](https://user-images.githubusercontent.com/36150943/68541316-582b9e80-03e1-11ea-91c2-d56decb4d597.png)

![image](https://user-images.githubusercontent.com/36150943/68541351-9fb22a80-03e1-11ea-91de-683455ba93b1.png)

![image](https://user-images.githubusercontent.com/36150943/68541422-a3927c80-03e2-11ea-9902-5728d6e29ee4.png)

![image](https://user-images.githubusercontent.com/36150943/68541432-b73de300-03e2-11ea-823a-80755771f490.png)


## Dependencies
* python 3.6
* keras
* numpy
* matplot
* opencv
* skimage

## Reference
1. B Lim et al. [Enhanced Deep Residual Networks for Single Image Super-Resolution.](https://arxiv.org/abs/1707.02921) CVPR 2018
2. Keras Subpixel (Pixel-Shuffle layer) from: [Keras-Subpixel](https://github.com/atriumlts/subpixel/blob/master/keras_subpixel.py)
3. DIV 2K image set from : [DIV_2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

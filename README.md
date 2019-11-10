# EDSR
* This page is a project for graduation from Hongik University in 2019. 
* EDSR(Enhanced Deep Residual Networks for Single Image Super-Resolution) with keras

![image](https://user-images.githubusercontent.com/36150943/68541243-57463d00-03e0-11ea-81d7-29c0299ac610.png)



## Data preprocess 
* The Datasets are DIV2K(you have to store the datasets in the "dataset" directory)
* Test images 800, validation images 100
* By using opencv libraries, the DIV2k Images are separated from 2k to patch(50x50), and the patchs are stored in the "processed" directory.
* Totally test patch images 49110, validation patch images 6250

## Train
After data preprocess finished, then you can train the model by using 'train' method, and the trained model stored the "models" directory as '.h5' file format.

```python
from edsr_model import EDSR_MODEL
ed = EDSR_MODEL()
ed.train()
```

## Predict
After training model finished, then you can process SR to any LR images.(the LR images you want to process have to be stored in the "test_image" directory

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

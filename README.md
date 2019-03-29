# A simple yet effective baseline for 3D human pose estimation

My own Gluon reimplement of [A simple yet effective baseline for 3D human pose estimation](https://arxiv.org/abs/1705.03098)</br>
Here is the [original implementation](https://github.com/una-dinosauria/3d-pose-baseline)</br>

## Enviroments
python3.6</br>
mxnet-gluon 1.4.0</br>
CUDA 9.0

## Dependency
``` 
pip install pyyaml
pip install scipy
pip install matplotlib
pip install easydict
``` 

## Dataset
[Here](https://pan.baidu.com/s/1Qg4dH8PBXm8SzApI-uu0GA) (code: kfsm) to download the HM3.6M annotation

## How-to-use
Specify your **root_path** and **dataset_path**
 in config.py  
And, you can modify configurations(like GPU) in config.py.  

**Train**: python train.py </br>

**Test**:  python test.py /path/to/model </br>
#TODO: provide trained model 

## Results
Since I don't have 2D pose estimate results on HM3.6M, I just experiment with 2D ground truth as input.
My best result is **46.2mm**(no augment is used), slightly higher than 45.5mm reported by paper.</br>

![Protocol1_Joint_Error](https://github.com/JimmySuen/Pose2DTo3D/blob/master/Baseline/doc/Protocol1_Action_error.png)

![Figure_1-1](https://github.com/JimmySuen/Pose2DTo3D/blob/master/Baseline/doc/Figure_1-1.png)

![Figure_1-5](https://github.com/JimmySuen/Pose2DTo3D/blob/master/Baseline/doc/Figure_1-5.png)

![Figure_1-9](https://github.com/JimmySuen/Pose2DTo3D/blob/master/Baseline/doc/Figure_1-9.png)

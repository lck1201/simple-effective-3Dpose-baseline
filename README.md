# A simple yet effective baseline for 3D human pose estimation

My own Gluon reimplement of [A simple yet effective baseline for 3D human pose estimation](https://arxiv.org/abs/1705.03098)</br>
Here is the [original implementation](https://github.com/una-dinosauria/3d-pose-baseline)</br>

- [ ] Todo: Provide trained model </br>

## Enviroments
python3.7</br>
mxnet-cu90 1.4.0</br>
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
```bash
usage: train.py/test.py [-h] --gpu GPU --root ROOT --dataset DATASET [--model MODEL]
                        [--debug DEBUG]

optional arguments:
  -h, --help         show this help message and exit
  --gpu GPU          number of GPUs to use
  --root ROOT        /path/to/code/root/
  --dataset DATASET  /path/to/your/dataset/root/
  --model MODEL      /path/to/your/model/, to specify only when test
  --debug DEBUG      debug mode
```

**Train**: python train.py --data /datapath --root /project-path --gpu /gpu-to-use </br>

**Test**:  python test.py --data /data-path --root /project-path --gpu /gpu-to-use --model /model-path </br>

PS: You can modify default configurations in config.py.

## Results
Since I don't have 2D pose estimate results on HM3.6M, I just experiment with 2D ground truth as input.
My best result is **46.2mm**(no augment is used), slightly higher than 45.5mm reported by paper.</br>

![Error](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/src/doc/Protocol1_Action_error.png)

![Figure1](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/src/doc/Figure1.png)

![Figure2](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/src/doc/Figure2.png)

![Figure3](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/src/doc/Figure3.png)

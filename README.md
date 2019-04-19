# A simple yet effective baseline for 3D human pose estimation

My own Gluon reimplement of [A simple yet effective baseline for 3D human pose estimation](https://arxiv.org/abs/1705.03098)</br>
Here is the [original implementation](https://github.com/una-dinosauria/3d-pose-baseline)</br>

Todo:
- [x] Provide trained model </br>
- [ ] Provide results on 2D pose estimates as input

## Enviroments
python 3.7</br>
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
1. [Baidu Disk](https://pan.baidu.com/s/1Qg4dH8PBXm8SzApI-uu0GA) (code: kfsm) or [Google Drive](https://drive.google.com/file/d/1wZynXUq91yECVRTFV8Tetvo271BXzxwI/view?usp=sharing) to download the HM3.6M annotation
2. Unzip data under *data* folder, and organize like this
```
${PROJECT_ROOT}
    `--data
        `--annot
            `--s_01_act_02_subact_01_ca_01
            `--s_01_act_02_subact_01_ca_02
            `-- ......
            `-- ......
            `-- ......
            `--s_11_act_16_subact_02_ca_04            
```

## How-to-use
You can download my trained model from [Google Drive](https://drive.google.com/file/d/1GvUCacxDBwrDsnWt1qcuhJNgtxcUWYeD/view?usp=sharing), which MPJPE is 44.9mm.

```bash
usage: train.py/test.py [-h] --gpu GPU --root ROOT --dataset DATASET [--model MODEL]
                        [--debug DEBUG]

optional arguments:
  -h, --help         show this help message and exit
  --gpu GPU          GPUs to use, e.g. 0,1,2,3
  --root ROOT        /path/to/project/root/
  --dataset DATASET  /path/to/your/dataset/root/
  --model MODEL      /path/to/your/model/, to specify only when test
  --debug DEBUG      debug mode
```

**Train**: python train.py --root /project-root --gpu /gpu-to-use </br>

**Test**:  python test.py  --root /project-root --gpu /gpu-to-use --model /model-path </br>

PS: You can modify default configurations in config.py. Because it's a quite simple system, not many hyperparameters need to be tuned.

## Results
Since I don't have 2D pose estimate results on HM3.6M, I just experiment with 2D ground truth as input.
My best result is **44.9mm**(no augment is used), slightly better than 45.5mm reported by paper.</br>

| Method | Avg      |  Direct |   Discuss   |  Eating   | Greet  | Phone  | Photo | Pose  | Purch  | Sitting   | SittingD   | Smoke   |  Wait   | WaitD   | Walk   | WalkT   |
| :-------- | --------: | --------:| :------: |--------: | --------:| :------: |--------: | --------:| :------: |--------: | --------:| ------: |--------: | --------:| ------: |------: |
| My Result | 44.9  | 36.8 | 43.5 | 40.5 | 43.0 | 46.2 | 54.7 | 40.0 | 43.6 | 52.9 | 59.7 | 44.2 | 44.5 | 45.0 | 34.6 | 37.3 |
| Paper     | 45.5  | 37.7 | 44.4 | 40.3 | 42.1 | 48.2 | 54.9 | 44.4 | 42.1 | 54.6 | 58.0 | 45.1 | 46.4 | 47.6 | 36.4 | 40.4 |



![Figure1](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/doc/Figure1.png)

![Figure2](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/doc/Figure2.png)

![Figure3](https://github.com/lck1201/simple-effective-3Dpose-baseline/blob/master/doc/Figure3.png)

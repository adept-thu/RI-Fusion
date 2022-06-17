# 项目名称 RI-fusion

### 版本：v1.0

----

## 环境依赖
- python=3.7
- pytorch=1.10.1
- CUDA11.3
- mmdetection3d >= v1.0.0rc3

## 环境安装
- 参考 https://mmdetection3d.readthedocs.io/zh_CN/latest/getting_started.html

----

## 基本操作

### 一、拷贝数据集
### 1、新建kitti 文件夹
新建文件夹路径: 
1、mmdetection3d/data/kitti
2、mmdetection3d/data/kitti/ImageSets

### 2、拷贝数据
数据集路径：/home/adept3090/sdb/dataset/KITTI 
路径下的 testing 和 training 文件夹
本地数据路径：mmdetection3d/data/kitti

### 3、下载并配置数据集分割模板
+ 下载数据集配置
```shell
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
```
+ 补充：可以根据训练需要，对其中的样本数进行修改

### 4、生成数据文件
```shell
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

### 二、训练
+ 多卡训练
``` shell
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 nohup tools/dist_train.sh configs/ 6 --work-dir ./works_dir/5.21_3dssd-RI_car_test>> ./out_dir/2022.5.21_3dssd-RI_car_test.out& 
```

### 三、生成测试结果 
+ 修改 config/_base_/datasets/kitti-3d-3classes.py(kitti-3d-car.py)文件 
+ (如果遇到生成的结果并非 test 数据集时,需进行此项修改)
```python
	test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_test.pkl',
```

+ 运行test.py 生成测试集检测结果
``` shell
python tools/test.py  configs//.py works_dir/5.12_/epoch_79.pth --format-only --eval-options 'pklfile_prefix=./test_result/dynamic_car' 'submission_prefix=./test_result/dynamic_car'
```
+ test.py  --gpu-id 参数，修改使用的GPU。

### 四、提交测试文件
+ 注册 kitti账号
+ 填写项目
+ 提交 test 生成的 .txt 文件的文件夹（以zip压缩包的形式）

----

## Tensorboard 使用
### 1、设置端口转发(服务器)
```
ssh -L 16006:127.0.0.1:6006 username@118.178.187.157
```
### 2、打开面板(本地)
```
tensorboard --logdir=/path/to/log-directory/ --port=6006
```
---

## RI-fusion 模块配置
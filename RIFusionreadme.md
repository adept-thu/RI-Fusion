# 组内readme撰写规范
## 项目用途
RI-fusion项目仓库（基于BEV和点云的融合检测，可以加一张图片）

## 项目数据集
### 项目数据集简介
kitti
该数据集的场景信息: 包括行人，自行车和汽车、

组成部分（大约几个区域，多少帧）、

数据类型: BEV、点云

### 项目数据集地址
`./data/kitti`
### 项目中数据集地址的引用
写明在那些文件的什么地方用到了数据集，便于后续引用和修改

config/pointpillars/pointpillars_with_img.py 10

...

## 安装信息
###环境
- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (如果你从源码编译 PyTorch, CUDA 9.0 也是兼容的。)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMdetection3D](https://mmdetection3d.readthedocs.io/zh_CN/latest/getting_started.html)

###安装步骤
**Step 0.** 

```shell
conda create -n open-mmlab python=3.7 pytorch=1.9 cudatoolkit=11.0 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip3 install -e .
```
具体参考MMdetection3D网址


```
账户下没有安装 tree
```


## 训练测试

###训练

训练配置
显卡型号：3090

显卡数量：8

训练运行时间：7h

```shell
cd mmdetection3d

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 nohup tools/dist_train.sh configs/pointpillars/pointpillars_with_img.py 8 --work-dir ./works_dir/5.21_3dssd-RI_car_test>> ./out_dir/2022.5.21_3dssd-RI_car_test.out&
```
按规定格式命名输出文件

###测试

测试配置
显卡型号：3090

显卡数量：1

训练运行时间：0.3h
```shell
cd mmdetection3d
python tools/test.py  configs/pointpillars/pointpillars_with_img.py works_dir/5.12_/epoch_79.pth --format-only --eval-options 'pklfile_prefix=./test_result/RI-fusion' 'submission_prefix=./test_result/RI-fusion'

```
根据权重文件位置修改路径


##可视化
可以放一张已有图片进行参考
曹振强同学处获取

## 目前成果 

| Dataset | benchmark | Params(M) | FLOPs(M) |               Download               |      Config      |
|:-------:|:---------:|:---------:|:--------:|:------------------------------------:|:----------------:|
|  KITTI  |           |           |          | [model](https:) &#124; [log](https:) | [config](https:) |
|         |           |           |          | [model](https:) &#124; [log](https:) | [config](https:) |

## 注意事项 


##引用

##作者联系方式（qq微信）

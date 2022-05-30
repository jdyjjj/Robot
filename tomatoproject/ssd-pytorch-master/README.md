## SSD：Single-Shot MultiBox Detector目标检测模型在Pytorch当中的实现
---

**2021年2月8日更新：**   
**加入letterbox_image的选项，关闭letterbox_image后网络的map一般可以得到提升。**

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [评估步骤 How2eval](#评估步骤)
7. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [ssd_weights.pth](https://github.com/bubbliiiing/ssd-pytorch/releases/download/v1.0/ssd_weights.pth) | VOC-Test07 | 300x300| - | 79.39

## 所需环境
torch == 1.2.0

环境配置见：[https://blog.csdn.net/weixin_44791964/article/details/106037141?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165171733716781683970792%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=165171733716781683970792&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-106037141.article_score_rank_blog&utm_term=torch&spm=1018.2226.3001.4450](https://blog.csdn.net/weixin_44791964/article/details/106037141?ops_request_misc=%7B%22request%5Fid%22%3A%22165171733716781683970792%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=165171733716781683970792&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-106037141.article_score_rank_blog&utm_term=torch&spm=1018.2226.3001.4450)



## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载ssd_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 利用video.py可进行摄像头检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在ssd.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": 'model_data/ssd_weights.pth',
    "classes_path": 'model_data/voc_classes.txt',
    "model_image_size" : (300, 300, 3),
    "confidence": 0.5,
    "cuda": True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 利用video.py可进行摄像头检测。  

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2ssd.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类**，示例如下：   
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8. 将utils.config的num_classes修改成所需要分的类的个数+1，运行train.py即可开始训练。

## 评估步骤
评估过程可参考视频https://www.bilibili.com/video/BV1zE411u7Vw  
步骤是一样的，不需要自己再建立get_dr_txt.py、get_gt_txt.py等文件。  
1. 本文使用VOC格式进行评估。  
2. 评估前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 评估前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在评估前利用voc2ssd.py文件生成对应的txt，评估用的txt为VOCdevkit/VOC2007/ImageSets/Main/test.txt，需要注意的是，如果整个VOC2007里面的数据集都是用于评估，那么直接将trainval_percent设置成0即可。  
5. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
6. 运行get_dr_txt.py和get_gt_txt.py，在./input/detection-results和./input/ground-truth文件夹下生成对应的txt。  
7. 运行get_map.py即可开始计算模型的mAP。

## mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

## Reference
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/kuhung/SSD_keras  

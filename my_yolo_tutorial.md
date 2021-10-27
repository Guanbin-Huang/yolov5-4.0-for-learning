# yolov5的性能
<div align = center>
    <img src = './imgs/yolo.jpg' width = '100%'>
    <h4>image: yolov5 performance</h4>

</div>

<br><br>

# yolo的新功能
- 模型系列化
- 分布式训练
- FP16模型存储与推理
- 断点续传
- 批量预测图像
- Tensorboard可视化训练
- onnx、coreml等导出
- 二阶分类器
- 模型集成预测
- 模型剪枝
- conv 与 bn 的融合
<br><br>
# yolov5训练和预测技巧
- 预处理
    - 图像归一化
    - 矩形推理缩放
- 数据增强
    - mosaic
    - CutMix
    - CutOut
    - MixUp
- 训练优化
    - warmup
    - 超参数进化
    - autoAnchor
    - EMA(exponential moving average)
    - 学习率的调整
        - 余弦退火衰退
    - 多尺度训练
    - 梯度累积
- 后处理
    - Merge NMS
    - TTA(Test Time Augmentation)
<br><br>
# 学习目标
- 基础知识
    - 目标检测任务说明
    - 常用数据集
    - 性能指标和计算方法

- 小数据集举例训练（VOC）

- 原理
    - yolo目标检测的基本思想
    - yolo网络框架和组件
    - yolo损失函数
    - yolo目标框回归与跨网络预测
    - yolo训练技巧

- 代码
    - 项目目录结构
    - 模型构建相关代码解析
        - 激活函数及其代码
        - 网络组件及其代码
        - detect组件及其代码
        - model类及其代码
    
    - 数据集创建相关代码解析
        - 矩形推理与letterbox代码
        - 数据增强原理与代码
        - 自定义数据集代码
        - 数据集相关类代码
        - dataloader相关代码

    - general.py代码解析
        - autoAnchor代码
        - AP计算代码
        - build_targets代码
        - loss计算代码
        - 非极大值抑制代码

    - 辅助工具代码解析
        - torch_utils 代码
        - experimental代码

    - yolov5直接运行的代码
        - detect.py
        - test.py
        - trrain.py

# yolov5 network 框架
<div align = center>
    <img src = './imgs/yolov5framework.jpg' width = '100%'>
    <h4>image: yolov5 framework</h4>
</div>

# 关于目标检测(object detection)
##  1. <a name=''></a>任务说明
<div align = center>
    <img src = './imgs/different_tasks.jpg' width = '100%'>
    <h4>image: 不同的任务</h4>
</div>

##  2. <a name='-1'></a>常用的数据集
- VOC 2007 和 VOC 2012
- MS COCO

<div align = center>
    <img src = './imgs/voc_intro.jpg' width = '100%'>
</div>
<br><br>
<div align = center>
    <img src = './imgs/coco.jpg' width = '100%'>
</div>

[coco link](http://cocodataset.org/)

- 80 个类别
- 超过50万类别的标注
- 平均每个图像目标数为7.2

##  3. <a name='-1'></a>性能指标
### 精度指标
<div align = center>
    <img src = './imgs/metrics.jpg' width = '100%'>
</div>

- 一个不完全合适的类比扔飞镖，5个目标
    - ref: [全面梳理：准确率,精确率,召回率,查准率,查全率,假阳性,真阳性,PRC,ROC,AUC,F1 - Theseus的文章 - 知乎](https://zhuanlan.zhihu.com/p/34079183)
    - precision: 你扔8次，你中了多少次(如果你扔了很多次，即使你最终5个都中了，但是你的精准率还是很差的)
    - recall: 5个目标你中了多少个（我不管你扔了多少次，只要你最后都中了就行）
    - 小结论： 所以说precision 和 recall 和不可兼得的。 不同的应用要权衡优先precision 或者 recall。
        - 要求recall高的：金融诈骗，允许有误报率，我不想放过任何一个可疑的诈骗，因为【代价高】
        - 要求precision高: 你得很确认的才给我做。你给我放进回收站里的可都得确定是垃圾，千万不能有正常邮件啊

    - FLOPS（floating-point operations per second）
<br><br>

####  3.1. <a name='PrecisionandRecall'></a>Precision and Recall
<div align = center>
    <img src = './imgs/TP_FP.jpg' width = '100%'>
</div>
<br><br>

- 第一列是TP FP 是precision
- 混淆矩阵描述的永远是预测的结果
    - TP T：预测对了，预测为P
    - FP F：预测错了，预测结果为P
    等等

    例子：猫狗分类，猫是正样本
    那么 如果我看见一只猫，却认为认为它是狗，则这个样本是FN。因为首先你预测错了，你预测成了狗

- precision:  预测的准不准
- recall：找得全不全
<br><br>

####  3.2. <a name='IOU'></a>IOU
<div align = center>
    <img src = './imgs/IOU.jpg' width = '100%'>
</div>

- 交并比
- IOU的进化
    - IOU -> GIOU -> DIOU -> CIOU
    - IoU_GIoU_DIoU_CIoU.ipynb
- 值域 [0, 1]
- 阈值
    - 例如 $IoU \geq Thres$ as TP
    - $IoU \leq Thres$ as FP
    - 如果连框都没有，但是其实是有物体的，那么就是FN
    - 我们忽略TN，因为我们不去预测负样本


<br><br>

####  3.3. <a name='APandmAP'></a>AP and mAP
<div align = center>
    <img src = './imgs/AP_and_mAP.jpg' width = '100%'>
</div>

- AP是怎么算出来的呢？
    - 我们刚才已经搞清楚了precision 和 recall是怎么算出来的了，他们各自是由FP 和 FN 去决定的，而你是FP 还是 FN 其实是由你设置的阈值来决定的。所以我们可以说，每一个阈值，我们就会有不一样的precision 和 recall（他们的值构成了一对点)。那么我将阈值从0 到 1 都选一遍，就可以得到很多个点。我就可以描绘出这个曲线了。（而这仅仅只是一个类而已）
    - 得到了曲线后，我们再去算出PR曲线下的面积，面积就是AP值。


- mAP
    - 将上面的步骤对每一个类都做一遍，求平均，我们就可以得到mAP

<div align = center>
    <img src = './imgs/mAP_table.jpg' width = '100%'>
</div>

<br>

- 在coco中
    - AP@.5 指的是 AP with IOU = 0.5
    - AP@.75 指的是 AP with IOU = 0.75
    - AP@[.5 : .95] 指的是 mean AP with IOU from 0.5 to 0.95 with step size of 0.05

    $$
    \begin{align}
        mAP_{coco} = \frac{mAP_{0.5} + mAP_{0.55} + ... mAP_{0.95} }{10}
    \end{align}
    $$

    - 快速过一下一些概念眼熟一下
    <div align = center>
        <img src = './imgs/loose_and_tight.jpg' width = '100%'>
    </div>

    <div align = center>
        <img src = './imgs/COCO_AP.jpg' width = '100%'>
    </div>

####  3.4. <a name='AP'></a>AP的计算
- ref: https://www.youtube.com/watch?v=FppOzcDvaDI&t=359s

- 步骤：
    1. 获取所有的bbox
    2. 按confidence降序排列所有bboxes
    3. 计算所有的precision 和 recall
    4. 画出PR 曲线
    5. 计算出PR 曲线下的面积
    6. 计算其他类的AP
    7. 在其他的IOU阈值下计算mAP

    - 详细请查阅 calculate_mAP.pdf

### 速度指标
    
- forward time ：包括前处理（图像归一化等），网络前传，后处理耗时（NMS等）
- FPS：每秒钟能够处理的图像数量
- FLOPS：处理一张图需要多少的浮点运算数量，跟具体的硬件没有关系。可以公平地比较不同算法之间的检测速度。

# 下载 yolo 跑一跑
1. 将项目clone到你的vscode里
    ``` bash
    git clone https://github.com/ultralytics/yolov5.git
    ```

2. 确保环境已经安装好了

3. 下载预训练权重并放在weights里面

4. 测试一下，运行跑一跑结果detect.py的结果
- 
    ```bash
    python detect.py --source ./data/images/ --weights weights/yolov5s.pt --conf 0.4
    ```
- 请确保你的图片在上面的路径下，或者你自己找对路径（学习的时候请创建跟老师一样的路径）

# 准备数据集
1. 下载voc
    - voc 官网: http://host.robots.ox.ac.uk/pascal/VOC/index.html
    - <div align = center>
        <img src = './imgs/download_voc.jpg' width = '100%'>
    </div>

    - <div align = center>
        <img src = './imgs/voc_screenshot.jpg' width = '100%'>
    </div>
    - voc 的数据集label格式是xml,我们要将其转换成txt

2. 生成训练集和验证集文件
    - 解压数据集
    ```bash
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    ```
    解压过后他们会在同一个文件夹里
    文件目录如下w
    -   ```bash
        |-- VOCdevkit
        |   -- VOC2007
        |       |-- Annotations
        |       |-- ImageSets
        |       |-- JPEGImages
        |       |-- SegmentationClass
        |       |-- SegmentationObject
        |-- VOCtest_06-Nov-2007.tar
        |-- VOCtrainval_06-Nov-2007.tar
        |-- get_voc_ubuntu.py


        接着执行脚本

        python get_voc_ubuntu.py # 将xml转换成txt，并将其移动到VOC文件夹里，且分好了train 和 val
        ```


- 执行完上面的脚本后，出现了VOC文件夹，目录结构如下
-   ```bash
    ./VOC
    |-- images
    |   |-- train
    |   `-- val
    `-- labels
        |-- train
        `-- val
    ```
- train.txt和2007_test.txt分别给出了yolov5训练集图片和yolov5验证集图片的列表，含有每个图片
的路径和文件名
- VOC/images文件夹下有train和val文件夹，分别放置yolov5训练集和验证集图片；VOC/labels文
件夹有train和val文件夹，分别放置yolov5训练集和验证集标签（yolo格式）

# 修改配置文件
1. 在data/ 里修改data/voc.yaml的路径
    ```yaml
    train: /datav/shared/dataset/VOC/images/train/ 
    val: /datav/shared/dataset/VOC/images/val/       
    ```
2. 创建models/yolov5s_for_voc2007.yaml，修改nc为20

# 开始训练
1. 训练命令
- vscode当前工作路径  
```bash
/datav/shared/hgb/yolov5

python train.py --data data/voc.yaml --cfg models/yolov5s_for_voc2007.yaml --weights weights/yolov5s.pt --batch-size 16 --epochs 200
```









    
    










# Reference
- [系统性比较好的yolov5讲解](https://www.bilibili.com/video/BV1NA411s7Ba?from=search&seid=11897837956125787800&spm_id_from=333.337.0.0)

- [Mean Average Precision (mAP) Explained and PyTorch Implementation](https://www.youtube.com/watch?v=FppOzcDvaDI&t=359s)
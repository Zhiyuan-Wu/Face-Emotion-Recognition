# 人脸表情识别实验-技术说明

武智源
wuzy14@mails.tsinghua.edu.cn

## **1、介绍**
&#160; &#160; &#160;&#160;这篇文章将介绍我们在人脸表情实验代码实现的使用方法。我们提供预训练的模型，介绍使用模型进行测试的方法；以及获得这些模型的训练方法。
&#160; &#160; &#160;&#160;本文中的所有示例代码都默认以Release为当前目录。
&#160; &#160; &#160;&#160;如果有任何问题，请联系作者。
## **2、环境**
&#160; &#160; &#160;&#160;我们在下面的环境中成功测试了所有代码，其它版本的环境应该也可以正常使用：

 - CentOS 7.2
 - Caffe
 - MATLAB R2016b
 - Python 2.7.5

&#160; &#160; &#160;&#160;读者可能需要根据具体环境配置编译Caffe及Caffe对MATLAB的接口、LIBSVM，编译方法请参考相关文档。
&#160; &#160; &#160;&#160;硬件方面，Caffe需要GPU支持，在训练过程中观察到了约8GB的显存占用，在测试过程中观察到了约1GB的显存占用，降低Batch Size可以显著降低显存占用。
## **3、测试**
&#160; &#160; &#160;&#160;我们提供了预训练的模型，使用该模型可以获得我们的测试提交结果。首先，到[百度网盘](http://www.baidu.com)下载模型和数据，覆盖相应的两个文件夹。
&#160; &#160; &#160;&#160;然后，修改`test.m`脚本首段的重要参数：
 - **switch_***: 是否启用对应的模型，1或0
 - **fusion_weight**: 简单带权融合中各个模型的系数，5维向量
 - **weights**: 使用的CNN模型，Caffe模型地址
 - **face_mean**: 使用的CNN模型输入的均值，Caffe均值文件地址
 - **test_path**: 所要测试的图片文件夹地址，图片都是jpg格式
&#160; &#160; &#160;&#160;将需要测试的图片放到一个目录下，使用下面的语句测试，测试结果将保存在`output_file`所指定的文件中：
```
$ matlab
>> test
```
## **4、训练**
&#160; &#160; &#160;&#160;这一部分旨在说明训练过程用到的一些代码脚本的作用，以及快速的获得一个训练好的模型的大致过程，由于数据随机分割、随机初始化、以及训练细节的不同可能使结果略有差异。读者可以参考这些代码探索更好的训练过程。
#### **4.1 数据准备**
&#160; &#160; &#160;&#160;训练过程用到了若干个数据库，包括CK+、JAFFE、KDEF、TFEID、自采数据库和FER2013，我们将这些数据库的原始版本上传到了[百度网盘](http://www.baidu.com)，这与上面提到的链接是完全相同的，下载模型和数据，覆盖相应的两个文件夹。
&#160; &#160; &#160;&#160;首先，使用下面的方法将图片数据的目录结构都变换为易于处理的`./Emotion/*.jpg`的形式，并使用数据集增强方法合并与扩充所有数据集，以及按照8:2的比例随机划分为训练集和验证集：
```sh
$ cd utils
$ python fer_csv2mat.py
$ matlab
>> fer_mat2imglist
>> distribute
>> data_aug
```
&#160; &#160; &#160;&#160;这将在`data/img/`下集合所有的数据图片，并以两个文本文件标明标签和训练、验证集划分。
#### **4.2 CNN**
&#160; &#160; &#160;&#160;我们对CNN进行了若干个阶段的细调，首先在FER2013两个测试集上进行前两个阶段的细调，然后在联合数据集上进行第三阶段的细调，首先将数据库转换为LMDB格式以加快训练速度：
```
$ caffe/build/tools/convert_imageset --shuffle --resize_height=224 --resize_width=224 data/fer2013/img/ data/fer2013/PublicTest.txt data/fer2013/fer_public_test_lmdb
$ caffe/build/tools/convert_imageset --shuffle --resize_height=224 --resize_width=224 data/fer2013/img/ data/fer2013/PrivateTest.txt data/fer2013/fer_private_test_lmdb
$ caffe/build/tools/convert_imageset --shuffle --resize_height=224 --resize_width=224 data/img/ data/train.txt data/train_lmdb
$ caffe/build/tools/convert_imageset --shuffle --resize_height=224 --resize_width=224 data/img/ data/test.txt data/test_lmdb
$ caffe/build/tools/compute_image_mean data/train_lmdb data/face_mean_aug.binaryproto
```
&#160; &#160; &#160;&#160;然后分别进行三个阶段的训练：
```
$ caffe/build/tools/caffe train --solver=model/solver_stage1.prototxt --weights=model/VGG_FACE.caffemodel --gpu 0
$ caffe/build/tools/caffe train --solver=model/solver_stage2.prototxt --weights=model/mul_iter_1000.caffemodel --gpu 0
$ caffe/build/tools/caffe train --solver=model/solver_stage3.prototxt --weights=model/mul2_iter_500.caffemodel --gpu 0
```
#### **4.3 特征描述子+SVM**
&#160; &#160; &#160;&#160;首先，使用`feature_extract.m`脚本进行特征提取：
```
$ matlab
>> cd utils
>> method = 'all';
>> feature_extract
```
&#160; &#160; &#160;&#160;程序将按照准备好的数据以及数据集划分，对`data/img/`下的图片提取HOG、Dense-SIFT、LPQ、LBP四种图像特征描述子。将生成`data/feature/HOG.mat`等，以及`HOG_transparam.mat`等，前者包含训练集、测试集的数据和标签，后者包含数据归一化所需要的均值和标准差。
然后，使用train_single来训练相应的SVM：
```matlab
$ matlab
>> alldata = 0;
>> savemodel = 1;
>> method = 'HOG';
>> g = 0.0625;
>> c = 16;
>> train_single
```
&#160; &#160; &#160;&#160;我们给出我们在验证集上粗略调整的参数，对不同的特征重复执行上述步骤以获得全部的四个SVM。读者可以考虑搜索更好的参数。
|method|g|c|
|:-:|:-:|:-:|
|HOG|0.0625|16|
|dsift|0.03125|4|
|LBP|0.25|16|
|LPQ|0.0625|32|
&#160; &#160; &#160;&#160;生成的模型将保存到`model/`下。此后，读者可以按照**3、测试**一节描述的方法使用已训练好的模型进行测试。
#### **4.4 模型融合**
&#160; &#160; &#160; &#160;为了方便对融合系数进行调整，首先使用训练好的模型和分类器计算样本的概率分布，然后使用简单的带权融合手段尝试不同的组合方式：
```
$ matlab
>> cd utils
>> compute_prob
>> weight = [2;1;1;1;1];
>> combine
```
&#160; &#160; &#160;&#160;到这里，读者大致可以在验证集上获得90%的总准确率，以及95%的HOG准确率。值得说明的是：

 - 一次测试的结果是可能带有偶然性的，需要重复、独立的执行这样的训练过程约5次（因为每次随机取了20%作为验证集）的结果，取平均以获得近似5-fold交叉验证结果。
 - 实验表明，使用全部数据进行训练可以获得更好的结果。我们用来提交的测试模型就是在全部数据上训练得到的。


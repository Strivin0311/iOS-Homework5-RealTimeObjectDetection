# iOS-Homework5-RealTimeObjectDetection

* 本程序可以通过屏幕摄像捕捉图片，对图片类别进行判断
* 类别和判断把握度显示在屏幕下方的标签上
* 本程序的分类器是在CIFAR10数据集上，通过pytorch建立的CNN模型学习得到的(具体代码详见**CNN_MLModel.py**)
* 因此仅支持CIFAR10数据集所含的10个分类标签：
  * 'plane', 'car', 'bird', 'cat', 'deer', 
  * 'dog', 'frog', 'horse', 'ship', 'truck'
* 同时测试集上的准确率有60%

# 更新日志

## 2025.3.29

1. 新增 ```/ultralytics/nn/newmodule/convnextv2.py``` 、 ```/ultralytics/nn/newmodule/cswomtransformer.py``` 、 ```/ultralytics/nn/newmodule/efficientformerv2.py``` 、 ```/ultralytics/nn/newmodule/efficientvit.py``` 、 ```/ultralytics/nn/newmodule/fasternet.py``` 、 ```/ultralytics/nn/newmodule/lsknet.py``` 、 ```/ultralytics/nn/newmodule/mobilenetv4.py``` 、 ```/ultralytics/nn/newmodule/repvit.py``` 、 ```/ultralytics/nn/newmodule/rmt.py``` 、 ```/ultralytics/nn/newmodule/swintransformer.py``` 、 ```/ultralytics/nn/newmodule/vanillanet.py``` 文件内关于模型的相关信息（论文地址和论文代码地址）以及这篇论文试图解决什么问题的QA。

## 2025.3.28

1. 新增 ```network.py``` 、```train.py``` 、```val.py``` 、```detect.py``` 文件，分别对应模型结构快速查看（network.py）、模型训练（train.py）、模型验证（val.py）、模型预测（detect.py）

2. 修改 **Ultralytics** 源码中部分文件，用以适应 **YOLO** 模型的魔改。具体而言，修改 ```/ultralytics/nn/tasks.py``` 内部分内容，其中修改部分均用 ```### 魔改部分修改 ###``` 进行标注。

3. 新增 **YOLO魔改模块** 12个，共计63种不同尺度的模型：  
    - StarNet
        - starnet_s1、starnet_s2、starnet_s3、starnet_s4
    - RepVit
        - repvit_m0_9、repvit_m1_0、repvit_m1_1、repvit_m1_5、repvit_m2_3
    - CSwomTransformer
        - CSWin_tiny、CSWin_small、CSWin_base、CSWin_large
    - EfficientViT
        - EfficientViT_M0、EfficientViT_M1、EfficientViT_M2、EfficientViT_M3、EfficientViT_M4、EfficientViT_M5
    - SwinTransformer
        - SwinTransformer_Tiny、SwinTransformer_Small、SwinTransformer_Base、SwinTransformer_Large
    - LSKNet
        - lsknet_t、lsknet_s
    - EfficientFormerV2
        - efficientformerv2_s0、efficientformerv2_s1、efficientformerv2_s2、efficientformerv2_l
    - ConvNeXtV2
        - convnextv2_atto、convnextv2_femto、convnextv2_pico、convnextv2_nano、convnextv2_tiny、convnextv2_base、convnextv2_large、convnextv2_huge
    - FasterNet
        - fasternet_t0、fasternet_t1、fasternet_t2、fasternet_s、fasternet_m、fasternet_l
    - VanillaNet
        - vanillanet_5、vanillanet_6、vanillanet_7、vanillanet_8、vanillanet_9、vanillanet_10、vanillanet_11、vanillanet_12、vanillanet_13、vanillanet_13_x1_5、vanillanet_13_x1_5_ada_pool
    - MobileNetV4
        - MobileNetV4ConvSmall、MobileNetV4ConvMedium、MobileNetV4ConvLarge、MobileNetV4HybridMedium、MobileNetV4HybridLarge
    - RMT
        - RMT_T、RMT_S、RMT_B、RMT_L

4. 魔改模块完成 **v8** 、 **v10** 、 **11** 模型适配，均能在 ```network.py``` 中运行。yaml文件中未提及的其他尺度模型，可根据实际情况在 ```魔改后模型参数记录文件.md``` 中查看，并在yaml文件中对应位置进行修改  
    - V8模型位置：```/ultralytics/cfg/modules/v8_Hacking/```
    - V10模型位置：```/ultralytics/cfg/modules/v10_Hacking/```
    - 11模型位置：```/ultralytics/cfg/modules/11_Hacking/```
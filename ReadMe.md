## Xray disease detector

  The repository contains a pipeline for training a multi-class detection model (7 classes) for the task of detecting various diseases in chest X-rays.

  

The tools used in this work were:

 - Pytorch, Pytorch-lightning, Numpy, Hydra, MlFlow

 - For training I chose EfficientDet model (https://github.com/rwightman/efficientdet-pytorch).
 - В файле Data_exploration.ipynb - ноутбук с EDA
  

## Couple output results

![alt text](results/results3.png)
![alt text](results/results2.png)
![alt text](results/results1.png)
![alt text](results/results4.png)
![alt text](results/results5.png)

They are not perfect, but we can see that the model has begun to understand the consepts. 

## During training optimized:

Focal loss for classes,

bbox_loss - optimizes distance between bboxes

Validation is done on 15 percent of the dataset using the mAP metric.

## Metrics: 
At the end of the training we can do eval and calculate sensitivity specificity metrics. Sensitivity and specificity are particularly suitable for medical data because they provide crucial insights into the performance of diagnostic tests or models, which is very important for making informed medical decisions.

  

**High Sensitivity**: A test with high sensitivity correctly identifies most of the patients who have the disease. This is crucial in medical scenarios where missing a positive case (false negative) could lead to serious consequences, such as delayed treatment or disease progression.

**High Specificity**: A test with high specificity correctly identifies most of the patients who do not have the disease. This is important to avoid unnecessary treatments, anxiety, and further invasive diagnostic procedures.

  

## How can the current pipelines be improved:

1) Try other EfficientDet configurations (there are more than 50 of them) or other models (Detr, Detectron, Yolo ). As well as other image sizes.

2) Take additional data from other datasets, mark up more photos by rare classes. In general, the task of training a good detector on 800 images for 8 classes is quite difficult. It is even possible to generate synthetic data.

3) Come up with a set of good augmentations, so far I have taken the most basic ones and only experimented a bit with augmentations.

4) Experiments with hyperparameters.

5) Trying to find pre-trained detectors on other medical data and take those as a starting point.


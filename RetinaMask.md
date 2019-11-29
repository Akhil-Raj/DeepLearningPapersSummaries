# **Understanding RetinaMask**

### **Introduction**

##### _**WHY**_

To match the accuracy of Single-Shot detectors, which are extremely important popular in applications where speed and computational resources are important design considerations, with two-stage detectors.

##### _**HOW**_

This is done by improving training for the state-of-the-art single-shot detectors, RetinaNet, in three ways: 

1. Adding a novel instance mask prediction head to the single-shot RetinaNet detector during training.
2. A  new  self-adjusting  loss  function  that  improves  robustness during training.
3. Including  more  of  the  positive  examples  in  training, even those with low overlap.

RetinaMask has same computation speed cost as original RetinaNet but is more accurate. Also, it is possible to evaluate just the detection part of the RetinaMask.

### **Model** 

RetinaMask is built upon RetinaNet by introducing three modifications the the baseline settings. The three modifications are :-

1. Best Matching Policy
2. Self-Adjusting Smooth L1 Loss
3. Mask Prediction Module

##### **Best Matching Policy**

In the bounding box matching stage, the RetinaNet policy is as follows. All anchor boxes that have an intersection-over-union (IOU) overlap with a ground truth object greaterthan 0.5, are considered positive examples. If the overlap is less than 0.4, the anchor boxes are assigned a negative label. All anchors for which the overlap falls between 0.4 and 0.5 are not used in the training

But some of the ground truth objects' aspect ratios and outliers, with one side much longer than the other. Thus, no anchor boxes can be matched to those according to RetinaNet strategy.

So, RetinaMask relaxes overlapping IOU threshold to get them. Best matching anchor with any non-zero overlap gives the best accuracy.

##### **Self-Adjusting Smooth L1 Loss**

Smooth L1 Loss was originally proposed in [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)(page 3) to make bounding box regression more robust by replacing the excessively strict L2 loss :- 

{fig1.png}

Smooth L1 loss function with parameter Beta as defined in RetinaMask paper :-

{fig2.png}

It is used to split axis in two parts: L2 loss is used for targets in range [0, Beta] and L1 loss beyond that to avoid over-penalizing outliers. Choice of control point(Beta) is usually done by hyper parameter search.

In Self-Adjusting Smooth L1 Loss, running minibatch mean and variance of absolute loss are recorded with momentum 0.9 to calculate the Beta. Beta is chosen to be equal to the difference between the running mean and running variance, and the value is clipped to a range :-

{fig3.png}

Clipping is used because running mean is unstable during training, as the number of positive examples in each batch is different.

##### **Mask Prediction Module**

Single-shot detection predictions are treated as mask proposals. After running  RetinaNet for bounding box predictions, we extract the top N scored predictions. Then, we distribute these mask proposals to sample features from the appropriate layers of the FPN according to the equation :-

{fig4.png}

where constant= 4, and w and h are width and height of the detection respectively.

we use the {P3,P4,P5,P6,P7}(having same definition as in [FPN](https://arxiv.org/pdf/1612.03144.pdf) and [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)) feature layers for bounding box predictions and {P3,P4,P5} feature layers for mask prediction.

##### **Overall Network**

![RetinaMask Network](https://raw.githubusercontent.com/chengyangfu/retinamask/master/arch.png)

Figure above shows a  high-level  overview  of  the model.   Following  the  [Feature  Pyramid  Network](https://arxiv.org/pdf/1612.03144.pdf) setting, extra layers are added (P6 andP7) and form top-down connections (P5,P4,  andP3).   The  bounding  box  classification  head  con-sists of 4 convolutional layers (conv3x3(256) + ReLU) and uses 1 convolution (conv3x3(number of anchors * numberof  classes))  with  point-wise  sigmoid  nonlinearities.    For bounding box regression, class-agnostic setting is adopted.   We  also  run  4  convolutional  layers  (conv3x3(256)+ ReLU) and 1 output layer (conv3x3(number of anchors* 4)) to refine the anchors.  Once the bounding boxes arepredicted, we aggregate them and distribute to the Feature Pyramid  layers,  as  discussed  above.   The  ROI-Align  operation is performed at the assigned feature layers,  yielding 14x14 resolution features, which are fed into 4 conse-quent  convolutional  layers  (conv3x3),  and  a  single  trans-posed  convolutional  layer  (convtranspose2d  2x2)  that  up-samples the map to 28x28 resolution.  Finally, a predictionconvolutional layer (conv1x1) is applied. We predict class-specific masks.

### **Training Points**

The anchor boxes span 5 scales and 9 combinations (3 aspect ratios [0.5, 1, 2] and 3 sizes [20, 21/3, 22/3]). The base anchor sizes range from 322 to 5122 on Feature Pyramid levels P3 to P7.  Each anchor box is matched to no more than one ground truth bounding box.The anchors that have intersection-over-union overlap with a ground truth box larger than 0.5 are considered positive examples. On the other hand, if the overlap is less than 0.4, such anchors are treated as negative examples. Then, we use the proposed best matching policy, as described in Section 3, which can only add positive examples.

For each image during training, we also run suppression and top-100 selection of the predicted boxes (the same processing as single-shot detectors apply during inference).  Then, we add ground truth boxes to the proposals set, and run the mask prediction module. Thus, the number of mask proposals is (100+Gt) during training.

##### **Loss**

 The final loss function is a sum of the three losses: 
 Box Classification Loss + Box Regression Loss + Mask Loss
 
Where ,

1. Box Classification Loss is taken as Focal Loss as in [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf) :- 

	{fig5.png}

2. Box Regression Loss is taken as Self-Adjusting Smooth L1 and limit the control point to the range[0, 0.11]
3. Mask Loss can be defined as the average binary cross-entropy loss as described [here](https://blog.zenggyu.com/en/post/2019-01-07/beyond-retinanet-and-mask-r-cnn-single-shot-instance-segmentation-with-retinamask/) :-

	{fig6.png}

### **Results**
###### *(Note that dataset used for training in the research paper is   COCO dataset, which provides bounding box and segmentation mask annotations. They follow common practice, using the COCOtrainval135k split(union of 2014train 80k and a subset of 35k images from 2014val 40k) for training and the minival(remaining 5k  images  from  2014val  40k)  for evaluation.)*

##### **Comparison to RetinaNet**


Per-class difference of the mean Average Precision :-

![RetinaNet vs RetinaMask](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/9d8747468f0fed8e335656d7fe9737e4dc21c798/9-Figure7-1.png)

Note  that  the toaster class,  whose  mAP decreases  by 7.9 points (from  28.9  to  21.0), has only 9 ground truth objects in the validation set.

<u>Qualitative results between RetinaNet and RetinaMask </u>:-

1. Improvement in class with large aspect ratios like
1.1. No multiple detections for Tie
1.2. Better recall for skies
2. Less false negatives
3. Less Failure cases

<u>comparisons of RetinaMask to RetinaNet on different  backbone  networks  and  input  resolutions(*on COCOtest-dev*)</u> :- 

RetinaMask shows better accuracy for all combinations of backbone network choices and resolutions. It shows 1.84 mAP and 1.3 mAP improvement on ResNet-50 and ResNet-101 at input scale of 800 compared to RetinaNet.

##### **Comparisons to the state-of-the-art methods**

See the implementation details [here](https://arxiv.org/pdf/1901.03353.pdf)(on page 8)

Comparison is done with state-of-the-art methods on COCOtest-dev. Compared to RetinaNet, RetinaMask based on ResNet-101-FPN is better by 2.6 mAP. Compared to Mask R-CNN, it shows 3.5 mAP improvement based on ResNet-101-FPN

##### **Comparisons with  Mask  R-CNN  on  instance mask prediction**

For Comparison with Mask R-CNN on mask prediction  using  ResNet-101  on  COCOminival, RetinaMask models are trained in a very similar fashion to the +e2e training in [MaskRCNN](https://arxiv.org/pdf/1703.06870.pdf). Mask R-CNN still shows better accuracy on mask prediction, but the difference is only around 1.2 mAP.

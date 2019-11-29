#**Understanding FCOS: Fully Convolutional One-Stage Object Detection**

###**Introduction**

##### _**WHAT**_

A fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation.


##### _**WHY**_

Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, the proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training and significantly reduces the training memory footprint. More importantly, it also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance.

##### _**HOW**_

Authors first reformulate object detection in a per-pixel prediction fashion. They make use of multi-level prediction to improve the recall and resolve the ambiguity resulted from overlapped bounding boxes in training. Finally, a “center-ness” branch is used, which helps to suppress the low-quality detected bounding boxes and improve the overall performance by a large margin.

###**Anchor-based vs Anchor-free object detection**

#####**Drawbacks of anchor-based detectors**

1. Detection performance is sensitive to the sizes, aspect ratios and number of anchor boxes.

2. Because the scales and aspect ratios of anchor boxes are kept fixed, detectors encounter difficulties to deal with object candidates with large shape variations, particularly for small objects.

3. In order to achieve a high recall rate, an anchor-based detector is required to densely place anchor boxes on the input image. Most of these anchor boxes are labelled as negative samples during training. The excessive number of negative samples aggravates the imbalance between positive and negative samples in training.

4. An excessively large number of anchor boxes also significantly increases the computation and memory footprint when commputing IoU scores between anchors and GT boxes.

#####**Anchor-free detectors**

The most popular anchor-free detector might be YOLOv1.Instead of using anchor boxes, YOLOv1 predicts bounding boxes at points near the center of objects. Only the points near the center are used since they are considered to be able to produce higher-quality detection. However, since only points near the center are used to predict bounding boxes, YOLOv1 suffers from low recall as mentioned in YOLOv2. As a result, YOLOv2 makes use of anchor boxes as well. Compared to YOLOv1, FCOS takes advantages of all points in a ground truth bounding box to predict the bounding boxes and the low-quality detected bounding boxes are suppressed by the proposed “center-ness” branch. As a result, FCOS is able to provide comparable recall with anchor-based detectors as shown in our experiments.

The family of detectors have been considered unsuitable for generic object detection due to difficulty in handling overlapping bounding boxes and the recall being low. In this work, both problems can be largely alleviated with multi-level FPN prediction. Moreover, together with the proposed center-ness branch, the much simpler detector can achieve even better detection performance than its anchor-based counterparts.

###**Model**

Model can be described in three steps:

1. Reformulating object detection in per-pixel prediction fashion.
2. Use of multi-level prediction to improve the recall and resolve the ambiguity resulted from overlapped bounding boxes in training.
3. Proposing "center-ness" branch, which helps suppress the low-quality detected bounding boxes and improve the overall performance by a large margin.

#####**Object detection in per-pixel prediction fashion**

For each location (x, y) on the feature map _F_<sub>i</sub>, we can map it back onto the input image as ([s / 2] + x*s, [s / 2] + y*s), which is near the center of the receptive field of the location (x, y). The detector directly views locations as training samples instead of anchor boxes in anchor-based detectors, which is the same as in FCNs for semantic segmentation.

Specifically, location (x, y) is considered as a positive sample if it falls into any ground-truth bounding box and the class label c<sup>∗</sup> of the location is the class label of B<sub>i</sub>(ith bounding box). Otherwise it is a negative sample and c<sup>∗</sup> = 0 (background class).

Besides the label for classification, we also have a 4D real vector t<sup>∗</sup> = (l<sup>∗</sup> , t<sup>∗</sup>, r<sup>∗</sup>, b<sup>∗</sup>) being the regression target for each sample. Here <sup>∗</sup>, t<sup>∗</sup>, r<sup>∗</sup> and b<sup>∗</sup> are the distances from the location to the four sides of the bounding box, as shown in in the following Figure :

![Pixel Level Segmentation](https://www.groundai.com/media/arxiv_projects/525642/x1.png)

<p align="center">[Fig1]</p>

If a location falls into multiple bounding boxes, it is considered as an ambiguous sample. In the next section, it will be shown that with the multi-level prediction, the number of ambiguous samples can be reduced significantly.

Formally, if location (x, y) is associated to a bounding box B<sub>i</sub>, the training regression
targets for the location can be formulated as,

<p align="center">[eqn 1]</p>

It is worth noting that FCOS can leverage as many foreground samples as possible to train the regressor. It is different from anchor-based detectors, which only consider the anchor boxes with a highly enough IOU with ground-truth boxes as positive samples. It may be one of the reasons that FCOS outperforms it's anchor-based counterparts. For the same reason, FCOS has 9× fewer network output variables than the popular anchor-based detectors with 9 anchor boxes per location.


Since the regression targets are always positive, we employ exp(x) to map any real number to (0, ∞) on the top of the regression branch.

_Loss function_ : Training loss function is defined as:

![Loss](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWs2dU7ZkUDchD8yc2J-vpm6lx5r5wze7fdnHqF7BilLWfEKBlzw)

<p align="center">[eqn 2]<p>

where L<sub>cls</sub> is the focal loss as in RetinaNet and L<sub>reg</sub> is the IOU loss as in UnitBox. N<sub>pos</sub> denotes the number of positive samples and λ being 1 in the research paper is the balance weight for L<sub>reg</sub>. The summation is calculated over all locations on the feature maps F<sub>i</sub> . 1<sub>{c<sup>∗</sup><sub>i</sub> >0}</sub> is the indicator function, being 1 if c<sup>∗</sup><sub>i</sub> > 0 and 0 otherwise.


_Inference_ : The inference of FCOS is straightforward. Given an input images, we forward it through the network and obtain the classification scores p<sub>x,y</sub> and the regression prediction t<sub>x,y</sub> for each location on the feature maps F<sub>i</sub>. We choose the location with p<sub>x,y</sub> > 0.05 as positive samples and invert Eq. (1) to obtain the predicted bounding boxes.

#####**Multi-level prediction with FPN for FCOS**

Here we show that how two possible issues of the proposed FCOS can be resolved with multi-level prediction with FPN :

1. The large stride (e.g., 16×) of the final feature maps in a CNN can result in a relatively low best possible recall(BPR, Upper bound of the recall rate that a detector can achieve). But even with a large stride, FCN-based FCOS is still able to produce a good BPR, and can be better than anchor-based detector RetinaNet. Moreover, with multi-level FPN prediction, the BPR can be improved further to match the best BPR the anchor-based RetinaNet can achieve.

2. Overlaps in ground-truth boxes can cause intractable ambiguity during training, i.e., w.r.t. which bounding box should a location in the overlap to regress?

    Following FPN, we detect different sizes of objects on different levels of feature maps which are produced by backbone's CNN feature maps, which can be seen in the following:

    ![FCOS](https://image.slidesharecdn.com/anchorfreedetection-190503092210/95/anchor-free-object-detection-by-deep-learning-28-638.jpg?cb=1556875518)

    <p align="center">[Fig. 2]<p>

    we directly limit the range of bounding box regression on the basis on the values of (l<sup>∗</sup>, t<sup>∗</sup>, r<sup>∗</sup>, b<sup>∗</sup>) for each location at all feature levels. Since objects with different sizes are assigned to different feature levels and most overlapping happens between objects with considerably different sizes, the multi-level prediction can largely alleviate the aforementioned ambiguity and improve the FCN-based detector to the same level of anchor-based ones. we share the heads between different feature levels, not only making the detector parameter-efficient but also improving the detection performance.

Different feature levels are required to regress different size range. As a result, instead of using the standard exp(x) in the final, we make use of exp(s<sub>i</sub>x) with a trainable scalar s<sub>i</sub> to automatically adjust the base of the exponential function for feature level P<sub>i</sub>, which improves the detection performance.

#####**Center-ness for FCOS**

There is a performance gap between FCOS and anchor-based detectors due to a lot of low-quality predicted bounding boxes produced by locations far away from the center of an object. To suppress these low-quality detected bounding boxes without introducing any hyper-parameters, we add a single layer branch, in parallel with the classification branch to predict the “center-ness” of a location (i.e., the distance from the location to the center of the object that the location is responsible for) as shown in Fig. 2.

Given the regression targets (l<sup>∗</sup>, t<sup>∗</sup>, r<sup>∗</sup>, b<sup>∗</sup>) for a location, the center-ness target
is defined as,

![Center-ness](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxUJpRmcKpujbX4jSv1TBAHBYQIOAP7lXcXY-vdZH8Suw2cbU)

<p align="center">[eqn 3]</p>

The center-ness ranges from 0 to 1 and is thus trained with binary cross entropy (BCE) loss. The loss is added to the loss function Eq. 2. When testing, the final score (used for ranking the detected bounding boxes) is computed by multiplying the predicted center-ness with the corresponding classification score. Thus the center-ness can down weight the scores of bounding boxes far from the center of an object. As a result, with high probability, these low-quality bounding boxes might be filtered out by the final non-maximum suppression (NMS) process, improving the detection performance remarkably. Center-ness branch is learned during the training of networks and does not need to be tuned.

###**Results**

experiments are conducted on the large-scale detection benchmark COCO. Following the common practice, the COCO trainval35k split(115K images) is used for training and minival split(5K images) is used as validation for the ablation study. Main results are reported on the test dev split (20K images) by uploading detection results to the evaluation server. Unless specified, ResNet-50 is used as the backbone networks and the same hyper-parameters as RetinaNet are used.

#####**FCOS vs anchor-based detectors**
 
On comparing FCOS with RetinaNet, both having exactly same settings, FCOS still compares favorably against the anchor-based detector(36.4% vs 36.1%) with much less design complexity and using only half of the memory footprint than the anchor-based ones.

#####**Comparing with state-of-the-art Detectors**

For the main results on test−dev split, Authors make use of scale jitter as in RetinaNet during the training and double the number of iterations. Other settings are same as RetinaNet.

With ResNet-101-FPN and ResNeXt-32x8d-101-FPN as the backbone, FCOS outperforms RetinaNeXt with the same backbone by 1.9% and 1.3% in AP, respectively. FCOS also outperforms other classical two-stage anchor-based detectors such as Faster R-CNN by a large margin(~5 AP).

Compared to the recent state-of-the-art one-stage detector CornerNet, FCOS also has 0.5% gain in AP. Also, FCOS detector achieved the performance with a faster and simpler backbone ResNet-101 instead of Hourglass-104 in CornerNet and except for the standard post-processing NMS in the detection task, FCOS detector does not need any other post-processing. In contrast, CornerNet requires grouping pairs of corners with embedding vectors

#####**Substituting Region Proposal Networks**

FCOS should be also able to replace the anchor boxes in Region Proposal Networks (RPNs) with FPN in the two-stage detector Faster R-CNN. Compared to RPNs with FPN, we replace anchor boxes with the method in FCOS. Moreover, we add GN into the layers in FPN heads, which can make training more stable. All other settings are exactly same as official code.

With the proposed center-ness branch, FCOS further boosts AR 100 and AR 1k respectively to 52.8% and 60.3%, which are 21% relative improvement for AR 100 and 3% absolute improvement for AR 1k over the official RPNs with FPN. FCOS improves AR 100 and AR 1k by more than 9% and 3%, respectively, if GN is applied to RPN with FPN too.

# **Understanding RetinaNet**

### **Introduction**

##### _**WHY**_

The extreme foreground-background class imbalance encountered during training of dense detectors is the central cause for one-stage detectors trailing the accuracy of two-stage detectors.

##### _**HOW**_

Authors propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Their novel "Focal Loss" focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training.

The loss function is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases.

Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples.

### **Focal Loss**

To understand Focal less, we should revisit some terms:

1. Cross entropy Loss: It's given as : 
    
    ![cross entropy loss](http://images1.programmersought.com/187/5d/5d62952fd4cb34c00808701e4277633b.png)
  
   One notable property of this loss, which can be easily seen in its plot, is that even examples that are easily classified (probabilty >> 0.5) incur a loss with non-trivial magnitude. When summed over a large number of easy examples, these small loss values can overwhelm the rare class.
    
2. Balanced cross-entropy Loss: It's a common method for addressing class imbalance. It introduces a weighting factor α ∈ [0, 1] for class 1 and 1 − α for class −1. For notational convenience, we define α<sub>t</sub> analogously  to how we defined p<sub>t</sub> . We write the α-balanced CE loss as:
    
    [img1]
    
3. Focal Loss: While α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples. Authors propose to reshape the loss function to down-weight easy examples and thus focus training on hard negatives.
    
      More formally, they propose to add a modulating factor (1 − p<sub>t</sub>)<sup>γ</sup> to the cross entropy loss, with tunable focusing parameter γ ≥ 0. Focal loss is defined as:
        
    [img2] 
    
    Focal Loss has two properties:
    
    3.1 When an example is misclassified and p<sub>t</sub> is small, the modulating factor is near 1 and the loss is unaffected. As p<sub>t</sub> → 1, the factor goes to 0 and the loss for well-classified examples is down-weighted.
    
    3.2 The focusing parameter γ smoothly adjusts the rate at which easy examples are down weighted. When γ = 0, FL is equivalent to CE, and as γ is increased the effect of the modulating factor is likewise increased
    
    In practice we use an α-balanced variant of the focal loss:
    
    [img3]
          
    ###**RetinaNet Detector**
    
    RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. The backbone is responsible for computing a convolutional feature map over an entire input image and is an off-the-self convolutional network. The first subnet performs convolutional object classification on the backbone’s output; the second subnet performs convolutional bounding box regression. 
    
![retinNet](https://cdn-images-1.medium.com/max/2400/1*0-GVAp6WCzPMR6puuaYQTQ.png)

#####**Components of RetinaNet**

1. FPN

    FPN(Feature pyramid network) is adopted as the backbone network for RetinaNet. FPN augments a standard convolutional network with a top-down pathway and lateral connections so the network efficiently constructs a
    rich, multi-scale feature pyramid from a single resolution input image, see Figure (a) and (b) above. Each level of the pyramid can be used for detecting objects at a different scale. FPN
    improves multi-scale predictions
    
2. Anchors
    
    Translation-invariant anchor boxes are used. The anchors have areas of 32<sup>2</sup> to 512<sup>2</sup> on pyramid levels P<sub>3</sub> to P<sub>7</sub>, respectively. At each pyramid level we use anchors at three aspect ratios {1:2, 1:1, 2:1}. For denser scale coverage, at each level we add anchors of sizes {2 0 , 2<sup>1/3</sup>, 2<sup>2/3</sup>} of the original set of 3 aspect ratio anchors. This improve AP in our setting. In total there are A = 9 anchors
per level and across levels they cover the scale range 32 - 813 pixels with respect to the network’s input image.

3. Classification Subnet

    The classification subnet predicts the probability of object presence at each spatial position
    for each of the A anchors and K object classes.

4.  Box Regression Subnet
    
    In parallel with the object classification subnet, we attach a small FCN to each pyramid level for the purpose of regressing the offset from each anchor box to a nearby ground-truth object, if one exists. The design of the box regression subnet is identical to the classification subnet except that it terminates in 4A linear
    outputs per spatial location.

###**Results**

#####**Dataset Used**

Results are presented on the COCO dataset. For training, COCO trainval135k(80k images from train and a random 35k subset of images from the 40k images val split).

For the main results, COCO AP on the test-dev split is used, which has no public labels and requires use of evaluation server.


#####**Comparison to state of the art**

Upon comparing to one stage detectors, on RetinaNet-(Resnet)101-FPN with image size 800, AP gain of 5.9(33.2 bs 39.1) was acheived with the closest competitor, DSSD, while also being faster.

Compared to recent two-stage methods, RetinaNet achieves a 2.3 point gap above the top-performing Faster R-CNN model based on Inception-ResNet-v2-TDM. Plugging in ResNeXt-32x8d-101-FPN as the RetinaNet backbone further improves results another 1.7 AP, surpassing 40 AP on COCO.


# Compution Vision Notes

# General training tricks
+ First overfit to small dataset, than generalize.
+ Do some intution based binary search of hyperparameters
+ Use visualization as much as possible to analysis.
+ Don't change lots of hyperparameters at the same time.


## Variation Inference Methods

### Metrics
+ Inception Score [IS]
+ Frichent Inception Score [FID]
  - Penalizes for lack of varity in image and rewards for better quality.

### Generative Adversarial Networks (GANS)
#### 1)Original GAN

#### 2)BigGAN [pdf]
+ Solves issue of using GAN for high-definition images upto (512x512)
+ Resuts have 100 ~ 200 % improvement from SA-GAN (IS:166.5; FID 7.4)
+ Larger batch size(2048), with large number of channels(64)
+ paper is really emperical with all details of there experiments.
+ Uses truncation trick, orthogonal regulization, direct skip connections, 
+ Uses different distribution for sampling latent variable (z) incomparition to test time.
+ Result is not reproducible, unless u work in deepmind or have infinite compute power.
+ They provide list of negative results they encountered.
+ Training dataset suffer mode collapse, class leakageand local object artefacts.

#### GAN Training Tricks
+ Interesting tricks from soumith github repo [https://github.com/soumith/ganhacks]

## Video Tracking

+ Most of paper formulate problem in a graph structure so that generic graph algorithms can be used.
+ As frame rate->infy, tracking becomes detectuon.
+ Generally two statge object detector, where first stage is object dtection using RCN and second stage is box regression of tracking box using some localization algorithm.
Metric
+ MOTA <code>$$(1- \frac{\sum(m_t + fp_t + mme_t)}{\sum_tg_t}$$</code>; m -> no of misses; fp -> false positives and mme -> mismaches.

### MOT Challenge Data Set
MOT -> multiple Object Tracking
Task
+ Find where object are
+ Track object inbetween frame
Why hard
+ object shape and size can change drastically
+ object occulusion
+ Crowd ID switch, object detector switch ID to different ID, if similar object is near by.

#### Interesting papers

Tracking using object detector.

##### Aug 2019 Philip el al, Tracking without bells and whistles(TWBaW) [pdf](https://arxiv.org/pdf/1903.05625.pdf)
+ Simpliest approach to tracking multpile objects in a video.
+ Used off-the-self object detection algorithms(Faster R-CNN with RESNET-101 and Feature Pyramid Network[FPN]) with some nice tricks (Motion model and reindentification algorithms.
+ Currently state-of-art results on MOTChallenge database
+ Kill detection if previous and new object box are near by, just like ANMS.
+ Very good for ID switch problem, have supervisied algorithm to decrease ID switching.
Pro
+ Don't need lots of label data.
Con
+
Tracktor++
+ Added Motion model and re-identification 
+ Model model
  + Enhanced correlated coefficent


##### Heterogeneous Association Graph Fusion for Target Association in Multiple Object Tracking [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8540450&tag=1)
+ Solve the problem of Detector fails
+ Fuse Detection and superpixel graph

###### Improvements to Frank-Wolfe optimization for multi-detector multi-object tracking (Leibniz University Hannover, Technical University Munich)

###### Motion Segmentation & Multiple Object Trackingby Correlation Co-Clustering (University of Mannheim, University of Freiburg, Max Planck Institute, Bosch Institute) (JCC)
+ combine pixel tracjectory with bounding box tracjectory

###### Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-identification (Tsinghua University)(MOTDT17)
+ Use kalman filter object tracker
+ ReID network to join network
+ Use R-FCN object detector

###### Multiple Hypothesis Tracking Revisited (Georgia Tech, Oregon State)(MHT_DAM)(Good survey paper)






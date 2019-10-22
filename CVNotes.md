# Compution Vision Notes

# General training tricks
+ First overfit to small dataset, than generalize.
+ Do some intution based binary search of hyperparameters
+ Use visualization as much as possible to analysis.
+ Don't change lots of hyperparameters at the same time.
+ saw tooth learning rate schedule is good for fixing saddle point stuck issues
+ cosine learning rate schedule (converges fater then atom with sgd with mommentum)


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

#### PU-GAN 3D GAN for upsampling [pdf]()
+ Use GAN for upsampling
+ Have a very unique architecture using feature extraction and up-down-up expansion unit and coordinate reconstructor, self attention unit.
+ Vey good start-of-the art results



#### Semi-supervised learning context
+ Main idea is to use consisitency losses.

##### 2) Pathak et al Learning Features by Watching Objects Move, CVPR'17 [pdf](https://people.eecs.berkeley.edu/~pathak/papers/cvpr17.pdf)
+ Used optical flow results as a ground truth to train a image segmentation task.
+ Learned visual representation is applied to object detection, semantic segmentation and action recognition, and they showed that the results are better then other transfering learning done using other unsupervised learning setups.
+ They start with the bounding box crops ad placed jitter on it, they looked directly on the center of the object.
+ Done extensive Validation 
  + is segmentation a good pretext task?
    + pretrain using supervised maks and them compare to other pretext tasks
    + if the learned features are good, it should require small amount of fine tunning
    + Later layer should be specialized to major tunning
  + Can it learn from limited data or nosiy mask?
    + Even with noisy mask model can still learn.
  + Low shot learning transfer
    + Good features should require less training data to fine tune.
    + if you have less data you get better results if you freeeze initial layers and only tune last two layers.
  + doing unsupervising learning from static data doesn't make sense as our world is dynmaic and human learn in dynamic setup.
+ Strength
  + Well written paper, clearly defined hypothesisl and done plentiful experiments to validate their hypothesis.
+ Weaknesses
  + Predicted masks would be more structured if they would have been generated from conv layers rather than FC layer
  + uNLC can't mask out inanimate objects and it assumes that every video only has one moving objects, which is not reasonable for many application.
+ Future directions
  + Reframe the pretext as unsupervised instance segmentatopn it its own task.
  + Use a more powerful segmentation model for the pretext task.
  + Use the generated masks as ground truth for another model 
  + Use more powerfull network to pretext task.
  
   
  ##### Arandjelovic ́ Look, Listen and Learn
  + Learn audio and visual correspondance.
  + Two submetworks, vision and audio subnetwork.
  + Data sampling :- random frame from video and randon audio from another random video
  + Data augenttion:- flipping + color saturation + other methods
  + Dataset:- Flickr SoundNet
  + Pre-trained imagenet weights also git good performance in case of audio network.
  + Good HeatMaps, which show  that the trained nwtwork can correlate sounds can the object, which produces it
  + Used Unlabeleddaga
  
  ##### Selfie [pdf](https://arxiv.org/pdf/1906.02940.pdf)
  + Try to emulate pretraiing task of Bert, bu masking few patches of the image and then classify the correct position of image patches from given patches from same Image
  + Used Attention polling (Transformer)
  + Generalized Bert to continuous input space.
  + Used non-overlapping patches
  + Used large patches
  + include dropout in ResNet50
  + Used cosine learning schedule
  + Perform muliple classification simultenously inorder to speed up computation.
  + Didn't compare their attention pooling in comparision to max pooling and average pooling.
  
#### Invariant Information Clustering for Unsupervised Image Classification and Segmentation[pdf](https://arxiv.org/pdf/1807.06653.pdf)[Slide][review]
+ Clustering 
+ Maximize mutual information 
+ claim to be STOA
+ Have seconding network to combat overclusting (novel approach)
+ Very good result on image segmentation

#### Constractive predictive coding henaff et al 2019
+ One of the STOA, same results as supervised learning methods, better then supervised learning in some cases.


  
## Domain Adaptation 

#### Deep Transfer Learning with Joint Adaptation Networks [pdf](https://arxiv.org/pdf/1605.06636.pdf)
+ Contribution
+ Addressing joint adaption
+ Decrease the shift in the joint distribution across domains and enables learning transfering
+ Two loss:- Domain loss and classification loss

#### Auto-Dial Automatic DomaIn Alignment Layers[pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Carlucci_AutoDIAL_Automatic_DomaIn_ICCV_2017_paper.pdf)
+ The authors provide a way to train a network which can adapt to different domains and performs better in different settings.
+ Introduce DAlayer  & entropy loss to domain adaptation problem
+ Prior work
  + Adaptive batch norm
  + MMD + regulaization (KL divergence)
+ DALayer
  + introduce cross domain baise (alpha)
  + Align source and target domain
  + learned alpha,learn different degree of alignment 
  Weakness
    - No clear directive why source and target distribution are mixed
    - No rescaling/shifting of normalized values, unlike the batch norm
  
  + It is interesting to note that just by adding entropy loss had a negative impact, while entropy loss combined with DA-layer perform better consistently.
  + 

#### Domain-Adversarial Training of Neural Networks [pdf]
+ Use H-divergence as a regulaizer
+ Minimize the source risk along with minimizes h-divergence, They learn the trade off between risk minimization and H-divergence minimization
+ For effective do- main transfer to be achieved, predictions must be made based on features that cannot discriminate between the training (source) and test (target) domains.
+ Use gradient reversal layer to learn not to discriminate the the training and test target domain.
+ Use gradient reversal layer to learn everything in one pass within standard framework of gradient descent.
+ Good paper

Type of Domain adaptation papers can be classify into these three techniques
1) Statistical Alignment
  + Easy to train/compute/understand
  + just a loss/ norm no major model changes
  + Almost no new parameters to add
  
2) Feature Adverserial alignment
  - Oscillations during training (Due to GAN branch)
  + learned alignment
  + learning less parameters then pixelwise approach
 
  
3) Pixelwise learning
  - Learn lot of parameters more hen adversierial type
  - Theses learn parameters are only supervised using pixelwise loss
  - problem with pixelwise approach is that it may ignore the semantics information in the picture
  + Their are both easy and hard to implement
  + Assuming if you can easy implement that then you can easly interpret

## Video Tracking

+ Most of paper formulate problem in a graph structure so that generic graph algorithms can be used.
+ As frame rate->infy, tracking becomes detectuon.
+ Generally two statge object detector, where first stage is object dtection using RCN and second stage is box regression of tracking box using some localization algorithm.
Metric
+ MOTA <code>$$(1- \frac{\sum(m_t + fp_t + mme_t)}{\sum_tg_t}$$</code>; m -> no of misses; fp -> false positives and mme -> mismaches, gt ground truth



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

+ ##### Aug 2019 Philip el al, Tracking without bells and whistles(TWBaW) [pdf](https://arxiv.org/pdf/1903.05625.pdf)
  + Simpliest approach to tracking multpile objects in a video.
  + Used off-the-self object detection algorithms(Faster R-CNN with RESNET-101 and Feature Pyramid Network[FPN]) with some nice tricks (Motion model and reindentification algorithms.
  + Currently state-of-art results on MOTChallenge database
  + Kill detection if previous and new object box are near by, just like ANMS.
  + Very good for ID switch problem, have supervisied algorithm to decrease ID switching.
  Pro
    + Don't need lots of label data.
  Con
    + Metric is wrong, may be helping them to increase their score.
  Tracktor++
  + Added Motion model and re-identification 
  + Model model
  + Enhanced correlated coefficent

+ ##### Heterogeneous Association Graph Fusion for Target Association in Multiple Object Tracking [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8540450&tag=1)
  + Solve the problem of Detector fails
  + Fuse Detection and superpixel graph
+ ###### Improvements to Frank-Wolfe optimization for multi-detector multi-object tracking (Leibniz University Hannover, Technical University Munich)
+ ###### Motion Segmentation & Multiple Object Trackingby Correlation Co-Clustering (University of Mannheim, University of Freiburg, Max Planck Institute, Bosch Institute) (JCC)
  + combine pixel tracjectory with bounding box tracjectory
+ ###### Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-identification (Tsinghua University)(MOTDT17)
  + Use kalman filter object tracker
  + ReID network to join network
  + Use R-FCN object detector
+ ###### Multiple Hypothesis Tracking Revisited (Georgia Tech, Oregon State)(MHT_DAM)(Good survey paper)

#### Modelling uncertainity using Hedged instance embedding [pdf](https://arxiv.org/pdf/1810.00319.pdf)
+ Heteroscedastic uncertainity
  + data dependent uncertainity
   + Depends on inputs to model 
+ Homoscedastic uncertainity
  + task-dependnt uncertainity
  + Varies between different tasks
  + Stays constant for different inputs
+ in Mnist the amount of uncertainity correspons to corruption in the image.
+ Uncertianity is self mismatch probability.
+ AP is bettwe way ti measure than KNN for uncertainity.




   
  









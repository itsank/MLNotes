# Compution Vision Notes

# General training tricks


## Variation Inference Methods

### Metrics
+ Inception Score [IS]
+ Frichent Inception Score [FID]
  - Penalizes for lack of varity in image and rewards for better quality.

### Generative Adversarial Networks (GANS)
#### 1)Original GAN

#### 2)BigGAN
+ Solves issue of using GAN for high-definition images upto (512x512)
+ Resuts have 100 ~ 200 % improvement from SA-GAN (IS:166.5; FID 7.4)
+ Larger batch size(2048), with large number of channels(64)
+ paper is really emperical with all details of there experiments.
+ Uses truncation trick, orthogonal regulization, direct skip connections, 
+ Uses different distribution for sampling latent variable (z) incomparition to test time.
+ Result is not reproducible, unless u work in deepmind or have infinite compute power.
+ They provide list of negative results they encountered.
+ Training dataset suffer mode collapse, class leakageand local object artefacts.

#### Training Tricks
+ Interesting trick from soumith github repo [https://github.com/itsank/ganhacks]

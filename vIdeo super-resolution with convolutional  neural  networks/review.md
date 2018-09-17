# [Review] Video Super-Resolution With Convolutional Neural Netowkrs

## Paper summary

### 1. Introduction
Image and video or multiframe super-resolution is the process of estimating a high resolution version of a low resolution image or video sequence. It has been studied for a long time, but has become more prevalent with the new generation of Ultra High Definition (UHD) TVs (3,840 x 2,048). Most video content is not available in UHD resolution or lower resolutions. 

Inspired by the recent successes achieved with CNNs [15], [16], a new generation of image SR algorithms based on deep neural nets emerged [17]–[21], with very promising performances.

In the classification and retrieval domains, CNNs have been
successfully trained on video data [23], [24]. Training for recovery purposes remains a challenging problem because the video quality requirements for the training database are high since the output of the CNN is the actual video rather than just a label.

Our main contributions can be summarized in the following aspects:
We introduce a video SR framework based on a CNN.
• We propose three different architectures, by modifying
each time a different layer of a reference SR CNN
architecture.
• We propose a pretraining procedure whereby we train
the reference SR architecture on images and utilize the
resulting filter coefficients to initialize the training of the
video SR architectures. This improves the performance of
the video SR architecture both in terms of accuracy and
speed.
• We introduce Filter Symmetry Enforcement, wich
reduces the training time of VSRnet by almost 20%
without sacrificing the quality of the reconstructed video.
• We apply an adaptive motion compensation scheme to
handle fast moving objects and motion blur in videos

Learning-based algorithms, although popular for image SR, are not very well
explored for video SR. In [14] a dictionary based algorithm is
applied to video SR. The video is assumed to contain sparsely
recurring HR keyframes. The dictionary is learned on the fly
from these keyframes while recovering HR video frames.

### 2. Deep Learning-Based Image Reconstruction
DNNs have achieved state-of-the-art performance on a num-
ber of image classification and recognition benchmarks, includ-
ing the ImageNet Large-Scale Visual Recognition Challenge
(ILSVRC-2012) [15], [16]. However, they are not very widely
used yet for image reconstruction, much less for video recon-
struction tasks. Research on image reconstruction using DNNs
includes denoising [31]–[33], impainting [31], deblurring [34]
and rain drop removal [35]. In [17]–[21], deep learning is
applied to the image SR task. Dong et al. [17] pointed out
that each step of the dictionary based SR algorithm can be re-
interpreted as a layer of a deep neural network. Representing an
image patch of size f × f with a dictionary with n atoms can
be interpreted as applying n filters with kernel size f × f on
the input image, which in turn can be implemented as a convo-
lutional layer in a CNN. Accordingly, they created a CNN that
directly learns the nonlinear mapping from the LR to the HR
image by using a purely convolutional neural network with two
hidden layers and one output layer. Their approach is described
in more detail in Section III-A.

Cheng et al. [20] introduced a patch-based video SR algo-
rithm using fully connected layers. The network has two layers,
one hidden and one output layer and uses 5 consecutive LR
frames to reconstruct one center HR frame. The video is pro-
cessed patchwise, where the input to the network is a 5 × 5 × 5
volume and the output a reconstructed 3 × 3 patch from the HR
image. The 5 × 5 patches or the neighboring frames were found
by applying block matching using the reference patch and the
neighboring frames. As opposed to our proposed SR method,
[18] and [20] do not use convolutional layers and therefore do
not exploit the two-dimensional data structure of images.
Liao et al. [21] apply a similar approach which involves
motion compensation on multiple frames and combining
frames using a convolutional neural network. Their algorithm
works in two stages. In the first stage, two motion compensa-
tion algorithms with 9 different parameter settings were utilized
to calculate SR drafts in order to deal with motion compensa-
tion errors. In the second stage, all drafts are combined using a
CNN. However, calculating several motion compensations per
frame is computationally very expensive. Our proposed adap-
tive motion compensation only requires one compensation and
is still able to deal with strong motion blur (see Figure 10).


### 3. Video Super-resolution with Convolutional Neural Network
#### A. Single Frame/Image Super-Resolution
Video SR model을 학습시키기 전에, 먼저 이미지에 대하여 pre-train 시켰음.
이미지에 대한 사전학습을 위하여 SR을 위한 모델을 사용하였으므로, 이후로는 이 모델을 reference model이라 부른다.
신경망 구조는 [learning a deep convolutional network for image super-resolution dong et. al.]에 제안된것을 사용한다. 이는 CNN 레이어만을 사용하기 때문에 입력 이미지의 크기에 상관없이 적용가능하며 patch-based algorithm이 아니라는 장점이 있다.

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/assets/dong.png" width="300">
</p>

마지막 레이어는 오직 1개 (흑백 이미지의 경우)의 채널 차원만을 가진다. 입력이미지 Y는 bicubic보간법으로 upsampled되어 들어가기 때문에 입력이미지와 출력이미지의 해상도는 같다. 이는 당시에는 일반적인 CNN으로는 upsampling하는것이 불가능하였기 때문이다. 분류문제와는 다르게 SR에서는 maxpooling이나 normalization layer와 같은 압축기법, 또는 계층은 사용하지 않는다. 모델은 이미지넷 데이터셋으로부터 추출된 패치들로 학습되었으며, 이렇게 추출된 데이터셋은 400,000장의 이미지로 구성된다.

### B. Video Super-resolution Architectures
비디오 SR을 위해서 목표 프레임에 이웃한 프레임들을 처리 과정에 포함시켰다. Figure 2는 논문에서 고려하는 입력 이미지들의 합병 시점에 따른 세가지 다른 구조를 나타낸다. 실제로는 3장의 이미지가아닌 그 이상의 이미지를 사용하는것도 가능하며 이에 대한 실험도 수행하였다.

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/assets/VSRCNN2.png" width="500">
</p>





## Discussion


## Reference
paper url : [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7444187](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7444187)
reference code : [https://github.com/flyywh/Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution)



## Author
Dohyun Kim
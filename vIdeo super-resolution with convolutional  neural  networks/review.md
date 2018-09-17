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
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20domain%20adaptive%20faster%20r-cnn%20for%20object%20detection%20in%20the%20wild/fig1.png" width="300">
</p>


Otherwise an additional layer with one kernel otherwise a postprocessing or
aggregation step is required. The input image Y is bicubically
upsampled so that the input (LR) and output (HR) images have
the same resolution. This is necessary because upsampling with
standard convolutional layers is not possible. A typical image
classification architecture often contains pooling and normal-
ization layers, which helps to create compressed layer outputs
that are invariant to small shifts and distortions of the input
image. In the SR task, we are interested in creating more image
details rather than compressing them. Hence the introduction of
pooling and normalization layers would be counter productive.
The model is trained on patches extracted from images from
the ImageNet detection dataset [38], which consists of around
400,000 images.




## Discussion


## Reference
paper url : [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7444187](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7444187)
reference code : [https://github.com/flyywh/Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution)



## Author
Dohyun Kim
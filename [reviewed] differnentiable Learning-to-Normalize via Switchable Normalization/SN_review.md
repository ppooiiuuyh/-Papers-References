# [review] Differentiable Learning-to-Normalize via Switchable Normalization


## 들어가기 앞서
**Paper URL**  : [https://arxiv.org/pdf/1806.10779.pdf](https://arxiv.org/pdf/1806.10779.pdf)

**Paper Info** : Ping Luo, Jiamin Ren, Zhanglin Peng

**References**
- [https://github.com/switchablenorms/Switchable-Normalization](https://github.com/switchablenorms/Switchable-Normalization) : Switchable nomarlization 저자 구현.
- [https://github.com/taki0112/Switchable_Normalization-Tensorflow](https://github.com/taki0112/Switchable_Normalization-Tensorflow) : 구현2
- [https://blog.lunit.io/2018/05/25/batch-instance-normalization/](https://blog.lunit.io/2018/05/25/batch-instance-normalization/) :Batch-IN reference
- [https://www.slideshare.net/ssuser06e0c5/normalization-72539464](https://www.slideshare.net/ssuser06e0c5/normalization-72539464) : Layer normalization slideshare
- [https://arxiv.org/pdf/1607.08022.pdf](https://arxiv.org/pdf/1607.08022.pdf) : IN Paper Reference
- [https://arxiv.org/pdf/1607.06450.pdf](https://arxiv.org/pdf/1607.06450.pdf) : LN Paper Reference
- [https://arxiv.org/pdf/1502.03167.pdf](https://arxiv.org/pdf/1502.03167.pdf) : BN Paper Reference

## Abstract (원문 번역)
우리는 DNN의 다른 normalization layer들을 위해 다른 연산을 선택하는것을 배우는 Switchable Normalization (SN)을 제안함으로써 Learning-to-nomalize 문제를 다룬다. SN은 means와 variances와 같은 통계요소들을 계산하기위하여 그들의 중요도 가중치에 따라 end-to-end방식으로 학습하는것을 통하여 channel, layer, minibatch를 포함하는 3가위위지 별개의 범주들을 중에서 전환한다. SN은 몇가지 좋은 특성들을 가지고있다. 첫째로, N은 다양한 네트워크 구조와 문제들에 적용될 수있다 (Fig.1 참조). 둘째로, 이는 넓은 범주(매우 작거나 큰경우의) batch size들에서도 강경하며, 작은 minibatch(예를들어 GPU당 2 image)를 사용할때에도 높은 성능을 유지한다. 세번째로, 그룹들의 수를 hyper-parameter로서 찾는 group nomalization 과는 달리 SN은 모든 channel들을 그룹으로 다룬다. 다른 부가기능 없이도, SN은 Image classification, object detecton과 segmentation, artistic image stylization, neural architecture search와 같은 다양한 도전적은 문제들에서 기존의 비교대상들의 성능을 뛰어넘는 결과를 보였다.우리는 SN이 deep learning에서 normlalization 기법의 효과를 이해하고 사용하는것을 쉽게 해주는데 도움이 되기를 바란다. SN의 구현 코드는 https://github.com/switchablenorms/ 에서 사용가능하다.

## 1.Introduction
Normalization기법들은 딥러닝분야, 특히 computer vision분야에서 매우 효과적으로 사용되고있습니다. 최근에는 batch normalization 이나 instance normalization, layer normalization과같은 기법들이 각각의 다른 네트워크구조들을 위해 개발되었고 좋은 결과를 보였습니다. 그런데 이런 성공에도 불구하고, 이를 이용한 기존 작업들은 수작업으로 이루어졌으며 대부분 모든 레이어에 걸쳐 같은 기법들을 적용하였습니다. 먼저 image classification 분야에 처음으로 적용되었던 batch normalization (BN) 부터 살펴보겠습니다. BN은 CNN의 hidden feature map들을 normalizing 하는 기법입니다. CNN의 feature map들은 Fig.1과 같이 minibatch size, num of channels, height and width of a channel들의 4차원 tensor (N,C,H,W)로 구성되어있습니다.

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/fig1.png" width="400">
</p>

몇가지 normalization 기법들을 살펴보면 다음과 같습니다.
- **Batch normalization** : BN 같은 경우에는 H, W 또한 마찬가지로 다른 mini batch의 샘플들(Batch)의 channle들을 평균내줌으로써 mean 과 variance를 계산합니다. 그리고 일반적으로 convolution layer 이후, activation layer 이전에 위치하여 mean과 unit variance를 0으로 만들어줍니다.

- **instance normalization** : IN은 주로 artistic image style transfer를 위하여 사요되었습니다.BN과는 다르게 IN은 샘플마다 각각 channel-wise방식으로 normalizing합니다.

- **Layer normalization** : LN같은 경우에는 RNN구조의 최적화를 위하여 사용되었습니다. CNN의 경우에는 잘 작동하지 않는다는 말도 있습니다. BN에서의 Batch size 부분을 뉴런으로 바꿔주기만 하면 사용가능합니다. BN에 비하여 batch size에 대한 의존도는 작습니다.

이와 같이 normalization기법들은 각각의 장단점이 있으며 각각의 특성에 맞추어 다른 분야에서 다르게 사용되어야합니다. 이는 구조를 설계하는것을 복잡하게 만듭니다. 또한 이거한 기법들은 일반적으로 모든 레이어에 걸쳐서 한가지 방식으로만 사용되며 이는 suboptimal한 결과를 나타낼 수 있습니다. 이를 해결하기위하여 저자들은 이들 기법을 결합하고 importance weight을 사용하여 선택하는 SN을 제안합니다.


Object Detection은 CNN의 발전에 힘입어 상당한 성능향상을 보여왔습니다. 하지만 여전히 training과 test데이터셋 사이에 상당항 domain shift를 일으킬수있는 view-points, object appearance, backgrounds, illumination, image quality, 등등 에서의 큰 변화에도 강경할 수 있는지는 도전적인 문제로 남아있습니다. 논문에서 사용한 예시와 같이 자율주행 차량의 경우를 생각해본다면, 학습할때 사용했던 이미지를 찍었을때의 카메라 상태와 주변 환경(날씨 등)은 실제로 주행할때의 상태와는 다를것입니다. 이러한 경우 'domain shift'가 발생하며 detection 성능을 저하 시킬 것입니다. Fig1을 통하여 실제로 각 데이터셋 별로 환경의 차이를 확인할 수 있습니다.(사실 저자들은 확인할 수 있다고는하는데 (c)는 단순히 각 기법을 선택한 비율이기 때문에 이것만으로는 이 선택이 최적의 선택인지는 확인할 수 없지 않은가 싶습니다.)

설계상 SN은 다양한 분야에 적용된 수 있습니다. Fig.1(C)와 같이 우리는 일반적으로 하나의 문제에서 한가지 normalization기법만을 사용하는 경우 suboptimal한 결과가 얻어진다는것을 확인할 수 있습니다.(마찬가지로 다양한 기법들 전부에서 좋은 효과를 보이는지는 알 수 없는것 같습니다.). 특히 image classification 이나 object detection의 backbone network에서는 골고루 선택되며, detection box head나 mask head 에서는 LN이, transfer같은 경우에는 IN이, RNN같은경우에는 GN이 주로 선택되는것을 확인할 수 있습니다. (이 역시 최적의 선택인지는 확신할 수 없으나 어느정도 기존 연구 결과들에 맞는것 같습니다). 또한 이렇게 선택적으로 normalization 기법을 사용함으로써 SN은 mini-batch size에 강경하게 됩니다.

본 논문의 conribution은 다음과 같습니다
1. 기존 normalization기법들을 통합하여 다양한 네트워크 구조와 문제들에 manually하게 기법들은 선택하지 않고 적용할 수 있도록 하였다. 특히 SN은 각 layer별로 기법을 선택할 수 있다
2. SN은 CNN,RNN/LSTM구조 모두에 적용할 수 있으며, image classification, object detection, artistic image stylization, LSTM을 이용한 neural architecture search들에서 성능 향상을 얻었다.
3. SN은 사용하고 이해하기 쉽기 때문에 기존에 hard craft된 기법들을 대체할 수 있을것으로 기대됩니다.

이러한 contribution이 가능하려면 우선 SN을 사용하였을 떄 정말로 기존 기법만들 사용했을 경우보다 성능이 개선되는지를 실험적으로 확인해 봐야겠습니다. 추가로 저는 개인적으로는 각 레이어에서 어떠한 기법이 선택되었는지를 확인하는것을 통하여 각 레이어의 특성을 파악하는대에도 효과적으로 사용될 수 있을것 같다는 생각이 들었습니다.

## 2. Switchable Normalization (SN)

#### 2.1 A General Form
SN을 살펴보기에 앞어서 일반적으로 normalization에 사용되는 기법의 형태를 살펴보겠습니다. CNN구조를 예로 들어보겠습니다. 이때  ![\mathcal{h}](https://latex.codecogs.com/gif.latex?h) 는 임의의 레이어에서의 input data이며 (N,C,H,W) 4D tensor로 이루어져 있습니다. 그리고 ![h_{ncij}](https://latex.codecogs.com/gif.latex?h_%7Bncij%7D) 와 ![\^{h}_{ncij}](https://latex.codecogs.com/gif.latex?%5C%5E%7Bh%7D_%7Bncij%7D)는 각각 normalization 전,후의 픽셀을 의미하며 이때, ![n \in [1,N] , c \in [1,C] , i \in [1,H] , j \in [1,W]](https://latex.codecogs.com/gif.latex?n%20%5Cin%20%5B1%2CN%5D%20%2C%20c%20%5Cin%20%5B1%2CC%5D%20%2C%20i%20%5Cin%20%5B1%2CH%5D%20%2C%20j%20%5Cin%20%5B1%2CW%5D)는 각각 픽셀의 index를 나타냅니다. ![\mu ](https://latex.codecogs.com/gif.latex?%5Cmu)와 ![\sigma ](https://latex.codecogs.com/gif.latex?%5Csigma)는 각각 평균과 표준편차를 의미합니다. 식은 다음과 같이 구성됩니다.

<!--
\hat{h}_{ncij}= \gamma \frac{h_{ncij}-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
-->

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Chat%7Bh%7D_%7Bncij%7D%3D%20%5Cgamma%20%5Cfrac%7Bh_%7Bncij%7D-%5Cmu%7D%7B%5Csqrt%7B%5Csigma%5E2&plus;%5Cepsilon%7D%7D&plus;%5Cbeta%20%5C%3A%5C%3A%5C%3A%20%281%29">
</p>

이때 ![\gamma](https://latex.codecogs.com/gif.latex?%5Cgamma) 와 ![\beta](https://latex.codecogs.com/gif.latex?%5Cbeta)는 각각 scale과 shiftparameter를 의미하며 ![\epsilon](https://latex.codecogs.com/gif.latex?%5Cepsilon)는 numerical instability를 방지하기위해 추가되는 매우 작은 상수를 의미합니다. 식 1은 각 픽셀들을 ![\mu ](https://latex.codecogs.com/gif.latex?%5Cmu)와 ![\sigma ](https://latex.codecogs.com/gif.latex?%5Csigma)를 이용해서 normlizing해주고 다시 이들의 representation capacity를 보존해주기 위해서 re-scale과 re-shift를 해줍니다. IN이든 LN이든 BN이든 기본적으로는 이 식을 따릅니다. 다만 어떠한 방식으로 ![\mu ](https://latex.codecogs.com/gif.latex?%5Cmu)와 ![\sigma ](https://latex.codecogs.com/gif.latex?%5Csigma)를 정할것인가만 다른것입니다. 이를 일반식으로 다시 표현해보면 다음과 같습니다.

<!--
\mu = \frac{1}{\left|I_{k}\right |}\sum_{(n,c,i,j)\in I_k}h_{ncij}, \: \sigma_k^2=\frac{1}{\left|I_k\right|} \sum_{(n,c,i,j)\in I_{k}}(h_{ncij}-\mu_{k})^2 \:\:\:\:  (2)
-->

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cmu%20%3D%20%5Cfrac%7B1%7D%7B%5Cleft%7CI_%7Bk%7D%5Cright%20%7C%7D%5Csum_%7B%28n%2Cc%2Ci%2Cj%29%5Cin%20I_k%7Dh_%7Bncij%7D%2C%20%5C%3A%20%5Csigma_k%5E2%3D%5Cfrac%7B1%7D%7B%5Cleft%7CI_k%5Cright%7C%7D%20%5Csum_%7B%28n%2Cc%2Ci%2Cj%29%5Cin%20I_%7Bk%7D%7D%28h_%7Bncij%7D-%5Cmu_%7Bk%7D%29%5E2%20%5C%3A%5C%3A%5C%3A%5C%3A%20%282%29">
</p>

이때 ![k\in\left\{in,ln,bn\right\}](https://latex.codecogs.com/gif.latex?k%5Cin%5Cleft%5C%7Bin%2Cln%2Cbn%5Cright%5C%7D) 이며 각 기법을 구분하기 위하여 사용됩니다. ![I_k](https://latex.codecogs.com/gif.latex?I_k)는 픽셀의 집합을, ![|I_k|](https://latex.codecogs.com/gif.latex?%7CI_k%7C)은 픽셀을의 수를 나타냅니다.

IN[[paper](https://arxiv.org/pdf/1607.08022.pdf)] : ![\mu_{in}, \sigma_{in}^2 \in \mathbb{R}^{N\times C}](https://latex.codecogs.com/gif.latex?%5Cmu_%7Bin%7D%2C%20%5Csigma_%7Bin%7D%5E2%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%20C%7D) 이며 ![I_{in} = \{(i,j)|i\in[1,H],j\in[1,W]\}](https://latex.codecogs.com/gif.latex?I_%7Bin%7D%20%3D%20%5C%7B%28i%2Cj%29%7Ci%5Cin%5B1%2CH%5D%2Cj%5Cin%5B1%2CW%5D%5C%7D) 이다. 즉 IN은 2NC element들을 가지며, mean value와 variance value는 N샘플의 각각에 대하여, 또 C채널에 각각에 대하여 (H,W)에 따라 결정된다. 쉽게말해서 instance 별로 (H,W)에 대하여 nomalization을 수행하였으니 이러한 mean value와 variance value가 N*C개 만큼 (N*C 차원) 있다는 뜻입니다. 당연한말입니다.

LN[[paper](https://arxiv.org/pdf/1607.06450.pdf)] : ![\mu_{ln}, \sigma_{ln}^2 \in \mathbb{R}^{N\times 1}](https://latex.codecogs.com/gif.latex?%5Cmu_%7Bln%7D%2C%20%5Csigma_%7Bln%7D%5E2%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%201%7D) 이며 ![I_{ln} = \{(c,i,j)|c\in[1,C],i\in[1,H],j\in[1,W]\}](https://latex.codecogs.com/gif.latex?I_%7Bln%7D%20%3D%20%5C%7B%28c%2Ci%2Cj%29%7Cc%5Cin%5B1%2CC%5D%2Ci%5Cin%5B1%2CH%5D%2Cj%5Cin%5B1%2CW%5D%5C%7D) 이다. 이경우에는 Layer-wise(chaanel-wise)로, 즉 CNN의 경우 하나의 이미지(instance)에 대하여 fillter (feature map)들 까지도 묶어서 normalization해주었으니 value들이 sample들 수, 즉 batch size만큼 나올것입니다.

BN[[paper](https://arxiv.org/pdf/1502.03167.pdf)] : ![\mu_{bn}, \sigma_{bn}^2 \in \mathbb{R}^{C\times 1}](https://latex.codecogs.com/gif.latex?%5Cmu_%7Bbn%7D%2C%20%5Csigma_%7Bbn%7D%5E2%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BC%5Ctimes%201%7D) 이며 ![I_{bn} = \{(N,i,j)|n\in[1,N],i\in[1,H],j\in[1,W]\}](https://latex.codecogs.com/gif.latex?I_%7Bbn%7D%20%3D%20%5C%7B%28N%2Ci%2Cj%29%7Cn%5Cin%5B1%2CN%5D%2Ci%5Cin%5B1%2CH%5D%2Cj%5Cin%5B1%2CW%5D%5C%7D) 이다. BN은 LN과 유사하지만 이경우에는 Layer-wise가 아닌 batch-wise로 수행하기 때문에 channel수 만큼 나올것입니다.

여기까지가 기존 기법들에 대한 review이며 아래에서는 본격적으로 SN기법이 어떻게 구성되는지를 살펴보겠습니다.


####2.2. Formulation of SN
SN은 아래와 같은 직관적인 수식으로 구성됩니다.
<!--
\hat h_{ncij}=\gamma\frac{h_{ncij}-\sum_{k\in\Omega}w_k\mu_k}{\sqrt{\sum_{k\in\Omega}w'_k\sigma^2_k+c}}+\beta
-->

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Chat%20h_%7Bncij%7D%3D%5Cgamma%5Cfrac%7Bh_%7Bncij%7D-%5Csum_%7Bk%5Cin%5COmega%7Dw_k%5Cmu_k%7D%7B%5Csqrt%7B%5Csum_%7Bk%5Cin%5COmega%7Dw%27_k%5Csigma%5E2_k&plus;c%7D%7D&plus;%5Cbeta%20%5C%3A%5C%3A%5C%3A%20%283%29">
</p>

여기서 ![\Omega](https://latex.codecogs.com/gif.latex?%5COmega) = {in,ln,bn} 이며, w와 w'는 각각 means와 variance에 대한 importance weight를 나타낸다. 각각의 경우에서 w와 w'는 scalar변수이며, 모든 channel간에 공유합니다. 따라서 3 (in,ln,bn) *2(w,w')개만큼의 imprtance weight가 있게됩니다. 중요한것은 ![\Sigma_{k\in\Omega}w_k=1,\Sigma_{k\in\Omega}w'_k=1, \forall w_k,w'_k\in[0,1]](https://latex.codecogs.com/gif.latex?%5CSigma_%7Bk%5Cin%5COmega%7Dw_k%3D1%2C%5CSigma_%7Bk%5Cin%5COmega%7Dw%27_k%3D1%2C%20%5Cforall%20w_k%2Cw%27_k%5Cin%5B0%2C1%5D)이며 이를 위해 아래와 같이 softmax처리 해준다는 점입니다.

<!--
w_k=\frac{e^{\lambda_k}}{\Sigma_{z\in\{in,ln,bn\}}e^{\lambda_z}},k\in\{in,ln,bn\} \:\:\: (4)
-->

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?w_k%3D%5Cfrac%7Be%5E%7B%5Clambda_k%7D%7D%7B%5CSigma_%7Bz%5Cin%5C%7Bin%2Cln%2Cbn%5C%7D%7De%5E%7B%5Clambda_z%7D%7D%2Ck%5Cin%5C%7Bin%2Cln%2Cbn%5C%7D%20%5C%3A%5C%3A%5C%3A%20%284%29">
</p>

![\lambda](https://latex.codecogs.com/gif.latex?%5Clambda)들은 softmax처리 하기전의 실제로 학습가능한 변수들입니다. 위에서 본것과 같이 ![\mu ](https://latex.codecogs.com/gif.latex?%5Cmu)와 ![\sigma ](https://latex.codecogs.com/gif.latex?%5Csigma)는 각 기법(statistics)들 별로 간단하게 구할 수 있습니다. 하지만 이 식들을 자세히 보면 어딘가 비슷한 부분이 많습니다. 즉 중복되는부분이 많아서 연산량에 낭비가 많이 발생하게됩니다. 본 논문의 저자들은 이러한 중복을 아래와같은 식의 재구서을 통하여 complexity를 ![O(NCHW)](https://latex.codecogs.com/gif.latex?O%28NCHW%29)만큼으로 줄이게됩니다. 이는 기존기법만을 사용할떄와 비슷한 수준입니다. 식은 아래와 같습니다.

<!--
\mu_{in}=\frac{1}{HW}\sum_{i,j}^{H,W}h_{ncij},\sigma^2_{in}=\frac{1}{HW}\sum_{i,j}^{H,W}(h_{ncij}-\mu_{in})^2, \\
\mu_{ln}=\frac{1}{C}\sum_{c=1}^{C}\mu_{in},\sigma^2_{ln}=\frac{1}{C}\sum_{c=1}^{C}(\sigma^2_{in}+\mu_{in})^2-\mu^2_{ln}, \\
\mu_{bn}=\frac{1}{N}\sum_{n=1}^{N}\mu_{in},\sigma^2_{bn}=\frac{1}{N}\sum_{n=1}^{N}(\sigma^2_{in}+\mu_{in})^2-\mu^2_{bn} \:\:\: (5)
-->

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cmu_%7Bin%7D%3D%5Cfrac%7B1%7D%7BHW%7D%5Csum_%7Bi%2Cj%7D%5E%7BH%2CW%7Dh_%7Bncij%7D%2C%5Csigma%5E2_%7Bin%7D%3D%5Cfrac%7B1%7D%7BHW%7D%5Csum_%7Bi%2Cj%7D%5E%7BH%2CW%7D%28h_%7Bncij%7D-%5Cmu_%7Bin%7D%29%5E2%2C%20%5C%5C%20%5Cmu_%7Bln%7D%3D%5Cfrac%7B1%7D%7BC%7D%5Csum_%7Bc%3D1%7D%5E%7BC%7D%5Cmu_%7Bin%7D%2C%5Csigma%5E2_%7Bln%7D%3D%5Cfrac%7B1%7D%7BC%7D%5Csum_%7Bc%3D1%7D%5E%7BC%7D%28%5Csigma%5E2_%7Bin%7D&plus;%5Cmu_%7Bin%7D%29%5E2-%5Cmu%5E2_%7Bln%7D%2C%20%5C%5C%20%5Cmu_%7Bbn%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cmu_%7Bin%7D%2C%5Csigma%5E2_%7Bbn%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28%5Csigma%5E2_%7Bin%7D&plus;%5Cmu_%7Bin%7D%29%5E2-%5Cmu%5E2_%7Bbn%7D%20%5C%3A%5C%3A%5C%3A%20%285%29">
</p>

그런데 저는 솔직히 이 식에서 분산이 어떻게 유도되는지 못따라가겠습니다. 심플한 예제로 대입해보면 되기는하더군요. 놀랍습니다!

**Discussion.** SN의 본질은 기존의 기법들을 하나로 통합하는것입니다. 이를 위한 매우 직관적인 방법중 하나는 모든 기법을 사용한 경우를 다 구해놓고 이들을 통합하는것입니다만 본 논문에 따르면 이러한 방법은 연산량이 너무 많아지기 때문에 적합하지 않다고합니다. 저는 추가로 그런방법이 잘 될지도 의문이긴합니다. 어쩃든 식(3)를 다시 보면 이는 결국 ![\gamma(w_{bn}(\frac{h_{ncij}-\mu_{bn}}{\sqrt{\Sigma_kw'_k\sigma^2_k+\epsilon}})+w_{ln}(\frac{h_{ncij}-\mu_{ln}}{\sqrt{\Sigma_kw'_k\sigma^2_k+\epsilon}})+w_{in}(\frac{h_{ncij}-\mu_{in}}{\sqrt{\Sigma_kw'_k\sigma^2_k+\epsilon}}))+\beta](https://latex.codecogs.com/gif.latex?%5Cgamma%28w_%7Bbn%7D%28%5Cfrac%7Bh_%7Bncij%7D-%5Cmu_%7Bbn%7D%7D%7B%5Csqrt%7B%5CSigma_kw%27_k%5Csigma%5E2_k&plus;%5Cepsilon%7D%7D%29&plus;w_%7Bln%7D%28%5Cfrac%7Bh_%7Bncij%7D-%5Cmu_%7Bln%7D%7D%7B%5Csqrt%7B%5CSigma_kw%27_k%5Csigma%5E2_k&plus;%5Cepsilon%7D%7D%29&plus;w_%7Bin%7D%28%5Cfrac%7Bh_%7Bncij%7D-%5Cmu_%7Bin%7D%7D%7B%5Csqrt%7B%5CSigma_kw%27_k%5Csigma%5E2_k&plus;%5Cepsilon%7D%7D%29%29&plus;%5Cbeta)과 같습니다. 즉 말 그대로 각 기법을 사용한 결과에 가중치를 두어 '선택'되는 효과를 준것입니다. 그런데 제 생각에서 앞의 직관적인 방법도 만약 효과만 있다면 (5)의 방법으로 연상량을 줄이면 해볼만 하지 않나 싶기도 합니다.

**Inference** IN과 LN의 경우에는 각 test sample마다 normalization을 수행합니다.반면에 BN은 batch단위로 수행하죠. 기존의 BN의 경우에는 moving average(각 mini batch별로 평균을 따로 구함)을 사용했다면 본 논문에서는 batch average를 사용한다고 합니다. 이 batch average는 다음의 두 단계로 이루어집니다. 첫번쨰로 네트워크의 모든 SN layer들의 parameter들을 freeze(비학습상태)시키고 임의로 선택된 몇몇 mini batch들을 feed해 줍니다. 두 번쨰로 이 결과로 나온 각각의 minibatch에서의 means와 variance들을 평균내어줍니다. 이렇게만 보면 기존의 방식과 같은것같다는 생각이 듭니다. 제 멋대로 이해해보면, 기존에는 학습과정에서 순차척으로 매 minibatch에 대한 mean과 variance를 moment를 적용해서 저장해두고 평가시에 사용합니다. 즉 moment에 따라 과거의 값이 중요해지거나 현재의 값이 중요해질 수 있겠습니다. 반면 batch average같은 경우에는 아예 평가시에 한번더 구해버리기 때문에 이러한 과거의 영향을 적게 받지 않나 싶습니다. 논문에서는 이러한 방식을 사용하면 기존 방식에 비하여 더 안정적인 결과를 얻을 수 있다고 하며 그에 대한 실험은 appendix에서 확인해 볼 수 있습니다.

**Implementation** SN은 Pytorch나 tensorflow를 사용하여 매우 쉽게 구현 될 수 있습니다. 역전파 연산도 이러한 프레임워크를 사용하면 자동으로 수행됩니다. 하지만 CUDA수준에서 직접 구현할 일이있다면 이 역전파 과정을 직접 구현해주어야 할텐데요, 이 역전파 식의 유도역시 appendix에 정리해 주셨습니다. 저자분들이 굉장히 성심성의껏 논문을 작성해주신듯합니다!

##3. Relations to Previous Work
**Normaliztion** 아래의 표는 SN과 BNmIN,LN,GN그리고 세개의 BN의 변형기법인 Batch Renormaliztion (BRN), Batch Kalman Normalizatino(BKN), Weight Normalization (WN)을 비교한 것입니다. 그리고 표를 보시면 SN기법이 연산량면에서도 비슷하고 범용성을 훨신 좋다는것을 알 수 있습니다.

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/table1.png">
</p>

표가 워낙 잘 정리되어있어서 따로 정리할 것은 없지만 논문의 내용중에서 흥미로운 부분만 따로 정리하겠습니다. 대부분의 경우에는 p나 epslion같은 hyper-parameter(대부분의 경우에는 안정성을 위해 사용되며 고정된 값을 가짐)를 사용하지만 WN의 경우에는 그렇지 않습니다. 이는 다른 기법들에 비해서 WN는 feature space보다는 network parameter들의 space에 있기 때문이라고 합니다. 또한 SN같은 경우에는 더 풍부한 statistics를 가지고 있음에도 연산복잡도는 비슷하다는 점이있습니다. 그런데 저는 이 statistic가 풍부하다는게 정확히 어떤의미이고 어떤점에서 좋은건지 잘 모르겠습니다.


## 4. Experiments
실험은 주로 다양한 시나리오에서 SN이 좋은 결과를 보인다는것 위주로 진행되었습니다. 원 논문에 상세하게 설명되어있는만큼 이부분은 생략하겠습니다.

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master%5Breviewed%5D%20/differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/asset/fig2.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/asset/fig3.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/asset/fig6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/asset/fig9.png">
</p>


## 5.implementation
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/%5Breviewed%5D%20differnentiable%20Learning-to-Normalize%20via%20Switchable%20Normalization/asset/fig7.png">
</p>
이는 저자분들이 논문에 올려주신 구현인데, 위와같이 연산의 중복성 제거 없이 각각에 대하여 직점 평균과 분산을 구하는 방식으로 되어있습니다.

## Author
Dohyeon Kim
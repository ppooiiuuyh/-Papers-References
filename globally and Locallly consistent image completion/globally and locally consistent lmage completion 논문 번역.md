# Globally and Locally Consistent Image Completion

## References
* paper link : http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
* https://jayhey.github.io/deep%20learning/2018/01/05/image_completion/

## 머릿말
이 번역은 개인 공부용, 또는 혹시 이 논문에 대한 한글번역이 필요하신 분들을 위해 틈틈히 진행하고있습니다. 역자의 영어실력이 미흡한 만큼 번역의 퀄리티가 좋지 못합니다. 또한 처음 공부하는 입장인만큼 잘못도니 부연설명으로 인해 오히려 혼란을 야기할수 있을것입니다. 이에 대하여 지적해주시면 수정하겠습니다.

## 1. introduction

image completion은  alternative contes들로 target region을 채울 수 있도록 하는 기법이다. 이는 원치않는  물체를 제거하거나 이미지 기반의 3D 재구성을 위하여 가려진 영역을 생성할 수 있도록 한다. patch-based image syntesis와 같은 많은 image completion에 대한 많은 접근법들이 제안되어왔음에도 불구하고, 이는 종종 high-level의 장면에 대한 recognition을 필요로 하기 때문에 여전히 도전적인 문제로 남아있다.

~~~
to-do: patch-based image syntesis 에 대하여 설명
~~~

(모델이) textured pattern을 완성해야 할 뿐만 아니라, anatomy(해부학적 구조)를 이해하는것 또한 중요하다. 이러한 관점을 기반으로, 이 연구에서 우리는 장면의 **local continuity** 와 **global composition** 둘 다를 하나의 프레임워크에서 고려하였다.

우리의 연구는 최근에 제안된 **Context Encoder (CE)**의 접근법 위에서 진행되었다.
~~~
Context encoder (CE)
Context Encoders: Feature Learning by inpainting. CVPR 2016
paper : https://arxiv.org/pdf/1604.07379.pdf
~~~

CE 접근법은 feature learning에 의하여 동기부여되었으며,어떻게 arbitarary inpainting mask를 어떻게 다루는지와 어떻게 고해상도 이미지에 기법을 적용시키는지에 대하여 완전하게 설명하고있디 않았다. 우리가 제안하는 접근법은 이 두 부분에 대하여 다루고있으며, 나아가 우리가 보여줄 결과와 같이 시각적 품질을 개선시켰다.
 우리는 fully convolutional network를 우리의 기본 접근법으로 사용하였고, 지역적으로도 전역적으로도 일치하는 결과를 내는 새로운 natural image completion을 위한 구조를 제안한다. 우리의 구조는 세가지의 네트워크로 구성된다.
 * completion network
 * global context discriminator
 * discriminator
 
Completion network는 fully convolutional하고 이미지를 완성시키기 위해 사용된다. 반면에 global, local context diciminator들은 학습을 위해 부가적으로 사용되는 보조 네트크이다. 이들 discriminator( global & local context discriminator) 들은 이미지가 일관적으로 (consistently라고 되어있는데 적당한 번역을 모르겠습니다) 완성되었는지를 판단하기위해 사용된다. Global discriminaator는 장면의 전역적 consistency를 인지하기 위해 전체 이미지를 입력으로 받는다. 반면에 local discriminator는 더 세부적인 모양의 퀄리티는 판단하기위해  단지 완성된(다시그려진, 채워진) 작은 영역 주변만을 본다. 매 학습 세대동안, discriminator들은 그들이 진짜와 완성된(채워진) 학습 이미지들을 올바르게 구분 했는지를 분간하기위해 먼저 업데이트된다.

~~~
Discriminator가 Generator보다 먼저 update 됩니다.
~~~

그 다음에, completion network가 disciminator network들이 구분 못할정도로 정교하게 비어있는 부분을 채울 수 있도록 업데이트된다. Fig.1에 보이는바와 같이, local, global context discriminator들을 함께 사용하는것은 진짜같은 이미지를 완성하는것에 중요하게 작용한다.
![](https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/Globally%20and%20Locallly%20consistent%20image%20completion/asset/fig1.png)

우리는 우리의 기법과 기존의 다른 기법들을 다양한 이미지를통해 비교분석 하였다. 우리는 또한 얼굴완성과 같은 더욱 도전적인 분야에 대한 결과도 나타내었는데, 우리의 기법은 눈, 코 , 입과 같은 단편적인 이미지들을 현실감있게 생성해 낼 수 있었다. 우리는 이 도전적인 얼굴완성분야에 대한 자연스러움(naturalness)의 정도를 user study를 통하여 평가하였는데, 실제얼굴과 우리의 결과의 차이를 77%나 분간해내지 못하였다.
요약하자면, 이 논문에서 우리는 다음을 나타낸다.
* a high performance network model that can complete arbutrary missing regions
* a globally and locally consistent adversarial training approach for image completion, and
* results of applying out approach to specific datasets for more challenging image completion

## 2. RELATED WORK
다양한 다른 기법들이 image completion task를 위해 제안되어왔다. 더 전통적인 기법들 중 하나는 diffusion-based image synthesis 이다. 이 기법은 target holes 주위의 local appearance를 그들을 메우는 방향으로 전파시킨다.
~~~
diffusion based image synthesis
비어있는 부분 주변에서 비어있는부분쪽으로 확산시켜가며 메우는 기법인것 같습니다.
~~~
![](https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/Globally%20and%20Locallly%20consistent%20image%20completion/asset/diffusion.jpeg)
(예시 그림)

예를들어, 전파는 isophote direction field로 (등광도선 방향? 뭔지 모르겠습니다) 행하여질 수 있고, 또는 local feature들의 histogram 들을 기반의 global image statistic을 기반으로 행하여질 수도 있다. 하지만 diffusion-based 접근법은, 일반적으로 오래된 사진등에서 발견되는 스크래치와 같은 작고 좁은 영역만을 채울수 있다.
Diffution-based 기법들과 대조적으로, patch-based 접근법들은 큰 영역을 메우는 더 복잡한 이미지 completion분제를 수행할 수 있다. Patch-based completion은 texture synthesis을 위해서 처음 소개되었는데 (1999년), 이 기법에서 texture patch들은 source image들에서 추출되었으며 target image에 복사된다.
이는 후에 image stitching (2003)과 energyoptimization 기반의 graph cuts, texture generation (2005) 으로 확장된다. Image completion을 위하여, 최적 patch 탐색과 같은 여러 변형기법들이 제안되었다. 특히, Wexler et al(2007)과 Simakov et al. (2008)은 더욱 일관성있게 메울수있는 global-optimization-based 기법을 제안하였다. 이러한 기법들은 후에 PatchMatch라 불리는 randomized patch search 알고리즘에 의하여 가속화되는데, 이는 실시간 고수준 이미지 편집을 가능하게 한다. Darabi et al. (2012)는 image gradients를 path들 간의 거리 메트릭에 통합시킴으로써 개선된 image completion을 발표했다. 하지만 이러한 기법들은 patch pixel valus들의 거리의 제곱합과 같은 저수준 feature들에 의존하고, 이는 복잡한 구조에서의 영역을 매우는것에는 비효율적이다. 나아가, 그들은 우리가 제안하는 기법과는 다르게 새로운 source image에 나타나지 않은 새로운 object를 찾아내는것은 불가능하다.
 large missing region을 생성하는 문제를 해결하기위해, structure guidance를 사용하는 몇몇 기법들이 제안되었는데, 이는 중요한 구조적 정보들을 보존하기 위해 일반적으로 수작업으로 구체화한다. 이는 흥미영역, 선, 곡선, 인지적 외곡등을 구체화 시키는것으로 행하여질수있다. 장면 구조에 대한 자공 추청을 위한 기법들 또한 제안되었다.
 * utilizing the tensor-voting algorithm to smoothly connect curves across holes
 * exploiting structure-based proiority for patch ordering
 * tile-based search space constraints
 * statistics of patch offsets
 * regularity in perspective planar surfaces

이러한 기법들은 중요한 구조를 보존함으로서 이미지 completion의 퀄리티를 개선시켰다.하지만, 이러한 guidances는 장면의 구체적 종류의 경험적인 제약에 기반하고, 그래서 특정 구조에 제한적이다.
ptach-based approach에 존재하는 명백한 가장큰 제약은 합성된 텍스쳐는 오직 입력 이미지에서만 얻을 수 있다는것이다. 이는 설득력있는 completion은 입력 이미지에 없는 texture를 요구할때 문제가된다. Hays 와 Efros (2007)는 large database of image를 사용한 image comletion 기법을 제안하였다. 그들은 우선 데이버테이스에서 가장 비슷한 이미지들을 찾고, 일치하는 영역을 match된 이미지에서 자르고 메우 ㄹ영역에 붙이는것으로서 이미지를 completion하였다. 하지만 이는 입력이미지와 비슷한 이미지가 데이터베이스에 있다는 가정이 필요로 하는데, 실제로는 그렇지 않을수도 있다는 문제가있다. 이는 정확히 같은 장면의 이미지들이 데이터베이스에 포함되어있는 특별한 경우에 대하여도 확장되었었다(2009). 하지만 정확히 같은 장면이 포함되어있어야 한다는 가정은 일반적인 접근과 비교해서 응용성이 너무 제한적이다.
사람 얼굴의 completion 또한 inpainting의 한 분야로서 주목받아 오고있다. Mohammed et al. (2009)는 얼굴 데이터셋을 사용하여 patch library를 구성하였고, global and local parametric model을 제안하였다. Deng etal.(2011)은 얼굴 이미지 복구를 위해 spectral-graph-based algorithm을 사용했다. 하지만 이러한 접근법은 patch들을 학습하기위해 정렬된 이미지를 필요로 하였고, arbitrary inpating problem에는 일반적이지 않다. Convolutional Neural Networks 또한 image completion을 위하여 사용되어왔다. 초기에, CNN-based image inpainting 접근법은 매우 작고 얇은 mask들에 제한적이었다.(2014). 비슷한 접근법들 또한 손실된 데이터 복구를 위해 MRI와 PET 이미지들에 적용되어왔다. 더욱 최근에, 우리의 연구와 동시에, Yank et al(2017) 또한 inpainting을 위한 CNN 기반의 최적화 접근법을 제안하였다. 하지만 우리의 접근법과는 다르게, 이방법은 모든 이미지에 대하여 최적화를 수행하기 때문에 컴퓨팅 시간을 증가시킨다.
 우리는 최근 제안된 Context Encoder(CE) [Pathak et al. 2016]을 바탕으로 연구하였는데, 이는 큰 영역에 대하여 CNN 기반inpainting을 적용시킬수있도록 확장시켰다. GAN의 원래 목적은 convolutional neral network들을 활용하여 Generative model들을 훈련시키는것이다 (원문 : the opriginal purpose of GAN is to train Generative models using convolutional neural networks). 이들 generator network들은 discriminator라 불리는 보조 네트워크를 사용하여 훈련되는데, discriminator는 generator에 의해 생성된 이미지가 실제이미지인지 분간하는 일을 한다. Generator 네트워크는 또 discriminator 네트워크를 교란시키기에 충분한만큼 정교한 이미지를 생성하도록 학습된다. Mean Squared Error (MSE) loss를  GAN loss와 함께 사용하는것을 통하여, Pathak et al[2016]은 MSE loss만을 사용할때 일반적으로 나타나는 번짐 현상을 피하면서, inpainting network가 128 x 128 pixel 이미지 중앙의 64 x64 pixel의 이미지를 메우도록 학습하는것을 가능하게 하였다. 우리는 fully convolutional network를 사용하여 arbitrary resolution에도 적용가능하도록 그들의 연구를 확장시켰고  global과 local discriminator를 함께 사용하는 방식을 사용하여 시각적 품질을 상당히 개선시켰다. GAN의 주요 이슈중 한가지는  학습중의 불안정성이다 [radford et al. 2016; salimans et al. 2016]. 우리는 순수하게 generative 모델을 학습시키지 않고, 안정성을 우선으로 학습절차를 튜닝하는것을통하여 이 문제를 회피하였다. 게다가 우리는 구조와 학습과정을 image completion 문제를 해결하기위해 매우 최적화하였다. 특히 , 우리는 하나가아닌 두개의 global dicriminator network와 local discriminator network를 사용하였다. 우리가 보여주는것과 같이, 이 둘을 같이 고려하는것은 결과에 상당히 중요한 영향을 미친다.
  우리의 접근법은 기존접근법들의 한계점을 극복하였고, 실제같이 다양한 장면들을 complete 하였다. Table1에는 다른 접근법들과의 고수준의 비교를 나타낸다. 힌편으로는 patch-based 접근법들은, 임의의 mask와 size에 대하여 양질의 재구성을 보여주었다.하지만 그들은 이미지 안에 나타나지 않은 새로운 단편이미지들을 제공가지 못하였고, 고수준의 의미에 대한 이해를 가지지 못하였다. 그들은 또한 local patch 수준에서의 유사성밖에 가지지 않았다 (local patch 수준의 일관성만 가지고있다는것같습니다). 반면에 context encoder-based 접근법은 새로운 object를 생성했지만, 고정된 저해상도 이미지에밖에 사용할 수 없다. 게다가, 그 접근법은 주변 영역의 연속성에 대한 고려가 없기 때문에 local consistency가 부족할 수 있다. 우리의 접근법은 arbitrary image size와 mask에 대하여도 적용할 수 있으며, 일관성있는 새로운 object들도 생성할 수 있다.

## 3.APPROACH

## Translator
Dohyun Kim

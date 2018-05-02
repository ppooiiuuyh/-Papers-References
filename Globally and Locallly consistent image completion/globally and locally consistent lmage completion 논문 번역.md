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
다양한 다른 기법들이 image completion task를 위해 제안되어왔다. 더 전통적인 기법들 중 하나는 diffusion-based image synthesis 이다.

## Translator
Dohyun Kim

# Domain Adaprive Faster R-CNN for Object Detection in the Wild


## 들어가기 앞서
**Paper URL**  : [https://arxiv.org/pdf/1803.03243.pdf](https://arxiv.org/pdf/1803.03243.pdf)
**Paper Info** : Yuhua Chen, Wen Li, Chistos Sakaridis, Dengxin Dai, Luc Van Gool (CVPR 2018)
**References**
- [http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html) : DANN 에 대하여 잘 설명해주신 사이트입니다.
- [http://theonly1.tistory.com/301](http://theonly1.tistory.com/301) : 마찬가지로 Domain adaptation 에 대하여 잘 설명해주신 사이트입니다.
- [https://arxiv.org/abs/1505.07818](url) : Domain-adverarial training of neural networks 논문입니다.


## Abstract (원문 번역)
Object detection은 일반적으로 training과 test data들이 identical distributed한 환경에서 뽑혔다고 가정하는데, 이는 실제로는 항상 가능한 일은 아니다. 그러한 distribution 불일치는 상당항 성능저하를 유발한다. 본 연구에서 우리는, object detection의 cross-domain robustness를 향상시키는것을 목적으로 한다. 우리는 domain shift를 두 수준으로 다룬다:

- the image-level shift : 이미지 스타일, illumination(광학요소), 등등
- the instance-level shift : 물체의 외양, 크기 등등

우리는 최근의 state-of-the-art 모델인 Faster R-CNN 을 기반으로 연구를 진행하였으며, domain discrepancy를 줄이기 위하여 image level과 instance level에서의 두가지로 domain adaptation component들을 설계하였다. 두 domain adaptation component들은 $$$ \mathcal{H}-divergence$$$ 이론에 기반하며, adversarial training 방식으로 domain classifier를 학습시키는것으로 구현되었다. 다른 level의 Domain classifier들은 consistency regularization에 의하여 더 강하게 Faster R-CNN의 region proposal network(RPN)을 학습하도록 강제될 수 있다. 우리는 우리가 제안한 새로운 접근법을 Cityscapes, KITTI, SIM10K, 등을 포함한 여러가지 데이터셋들을 사용하여 평가하였다. 결과는 우리가 제안한 기법이 다양한 domain shift 시나리오에도 강경하게 object detection을 수행할 수 있음을 보여주었다.

## 1.Introduction
Object Detection은 CNN의 발전에 힘입어 상당한 성능향상을 보여왔습니다. 하지만 여전히 training과 test데이터셋 사이에 상당항 domain shift를 일으킬수있는 view-points, object appearance, backgrounds, illumination, image quality, 등등 에서의 큰 변화에도 강경할 수 있는지는 도전적인 문제로 남아있습니다. 논문에서 사용한 예시와 같이 자율주행 차량의 경우를 생각해본다면, 학습할때 사용했던 이미지를 찍었을때의 카메라 상태와 주변 환경(날씨 등)은 실제로 주행할때의 상태와는 다를것입니다. 이러한 경우 'domain shift'가 발생하며 detection 성능을 저하 시킬 것입니다.

<p style="text-align: center;">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/-Papers-References/master/domain%20adaptive%20faster%20r-cnn%20for%20object%20detection%20in%20the%20wild/fig1.png" width="300">
</p>


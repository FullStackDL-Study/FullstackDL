# convnet architecture

## ALEXNET
ILSVRC에서 1등을 한 최초의 CONVNET

## ZFNet
- Alexnet은 계산과정이 오래걸리기 때문에, 수많은 Hyper Parameter를 최적화 하는데 큰 어려움이 있다.
- 이러한 단점을 극복하기 위해서 Alexnet에 Visualizing을 도입한 CONVNET

![img1](https://user-images.githubusercontent.com/84113554/120083701-8aa50580-c105-11eb-8ea6-8cf4bbfed053.png)

## VGG
- NN의 깊이가 성능에 미치는 영향을 확인하고자한 연구(깊이의 영향만을 확인하고자 Filter Size를 3 by 3으로 고정)
- 심플하지만 deep한 구조 => NN가 Deep 할수록 성능이 좋아진다는 것을 보여줌

![img2](https://user-images.githubusercontent.com/84113554/120083830-7ca3b480-c106-11eb-8da0-c14ede7dada4.png)
- 필터커널의 size를 3 by 3으로 한 경우 5 by 5로 한 경우보다 parameter의 수가 줄어든다.
- training 시 속도 ↑, 층이 많아지므로 비선형성을 더 잘나타낸다.
- 그럼에도 불구하고 너무 많은 parmeter의 수가 문제

## GoogLeNet
- VGG만큼 Deep하지만 Parameter는 훨씬 적음(3%)
- 일반적으로 Convnet은 deep할수록 성능이 좋아짐 하지만 Learning parameter가 많아지는 문제 발생
- 특히, Deep한 곳에 위치한 convolution 연산의 문제가 발생
- 연산량을 줄이기 위해서 1 by 1 convolution을 먼저 계산
![img3](https://user-images.githubusercontent.com/84113554/120084055-1e77d100-c108-11eb-8cff-98ba2043600e.png)
- weight의 수를 줄이기 위해서 Global average pooling 방식을 채택
- Vanishing Gradient 문제를 해결하기 위한 Auxiliary classifier를 추가(learning 시에만 작동)
![img4](https://user-images.githubusercontent.com/84113554/120084084-7adaf080-c108-11eb-84c2-e8ea2b817954.png)

## ResNet
- Deep한 모델에서 Vanishing Gradient 문제로 인해서 성능이 좋아지지 않는 문제
![img5](https://user-images.githubusercontent.com/84113554/120084137-e624c280-c108-11eb-8ac7-104dcec1a732.png)
- residual block을 추가

# Localization, Detection, Segmentation
다양한 object가 있을때는 localization이 되지 않는 문제를 다양한 scale에서 분류기를 슬라이드하는 것으로 해결(계산과정 ↑)

- FC의 경우 Fixed-Scale에서만 적용이 가능하지만, Convolution은 크기가 상관없기 때문에 1 BY 1 CONVOLUTIUON을 통해 다양한 사이즈에서 활용
(FC와 1 by 1 Convolution은 개념적으로 동일)
![image](https://user-images.githubusercontent.com/84113554/120084261-af9b7780-c109-11eb-9c2a-bb31bc4a3711.png)

- IOU(Intersection over Union) : Overlapping Region / Combined Region
- NMS(Non-Maximum-Suppression) : 겹치는 bounding box 중에 가장 좋은 Box를 선택하는 기법

# R-CNN
- Image Classification(CNN) + Localization(Region proposal)

# Faster R-CNN
- Selective search의 경우 CPU에서 동작하기 때문에 느리다.
- Region proposal을 생성하는 네트워크를 GPU에서 동작하기 위해 Conv layer에서 생성

# MASK R-CNN
- Image segmentation을 위한 모델

# Upsampling
- Unpooling : Max pooling pixel을 기억한 뒤, upsampling할 때 사용

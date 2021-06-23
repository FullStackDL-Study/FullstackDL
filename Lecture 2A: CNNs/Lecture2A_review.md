# Lecture2A: CNNs
## 1. Convolutional Filters
합성곱 필터(커널)은 두가지 이유 때문에 도입되었습니다. 첫째는 fully connected network의 경우 학습할때 매우 큰 행렬이 들어가므로 사진 크기를 키워주어야 하고(?) 사진을 키우면 더 많은 파라미터들을 결정해야만 하게 됩니다. 더 많은 weight때문에 성능에 안 좋은 영향을 미칩니다.

![3](https://user-images.githubusercontent.com/63699718/119652853-96709d80-be61-11eb-9e2f-9660b10dc85f.PNG)

둘째는 fully connected network를 사용하면 이미지내에서 중요한 부분과 아닌 부분을 똑같이 취급해서 처리하게 되기 때문입니다. 

필터는 이미지의 특징을 찾아내기 위한 공용 파라미터입니다. 이 커널은 입력 데이터와 합성곱 연산을 수행합니다. 또한 이를 통해 결과 데이터의 차원이 줄어듭니다. 

![58780750-defb7480-8614-11e9-943c-4d44a9d1efc4](https://user-images.githubusercontent.com/63699718/119650876-4b558b00-be5f-11eb-925c-f95bad9693d2.gif)

위 그림은 필터를 적용시키는 걸 보여줍니다. 여기서 패딩과 스트라이드의 개념이 나오는데 이는 아래에 서술합니다.

## 2. Filter Stacks and ConvNets
input과 output은 여러 channel를 가질 수 있습니다. 여기서 channel이란 Convolution Layer에 유입되는 입력 데이터에는 한 개 이상의 필터를 말합니다. 1개 필터는 Feature Map의 채널이 됩니다.
Convolution Layer에 n개의 필터가 적용된다면 출력 데이터는 n개의 채널을 갖게 됩니다.

## 3. Strides and Padding
스트라이드란 한 번 합성곱 연산한 후 다음 계산 영역을 선택할때 얼마나 이동할지 간격을 정하는 값입니다. 위 사진에서는 stride가 1인 경우 였습니다. 근데 만약 스트라이드가 3쯤이되어서 합성곱하는 부분이 그림을 벗어나게 되면 계산하는데 문제가 생깁니다. 그래서 이를 해결하기위해 padding이 등장합니다. 패딩은 출력 데이터의 차원을 키워주는 방법입니다. 보통 0을 가장자리에 집어넣어 해결합니다. 그럼으로써 스트라이드가 3이여서 생기는 문제도 해결 할 수 있습니다. 

![4](https://user-images.githubusercontent.com/63699718/119655817-f3ba1e00-be64-11eb-9988-6988e12d56c3.PNG)

## 4. Filter Math
한 합성곱의 output의 size가 무엇인지 어떻게 계산할 수 있을까요. 이는 뉴럴넷을 설계할때 그전 layer의 size가 다음 layer를 만들때 영향을 끼치기에 중요한 질문입니다. 이에 영향을 끼치는 파라미터들은 필터의 개수(stack), 스트라이드, 패딩으로 이것들은 중요합니다. 

합성곱의 수학적 배경은 A guide to convolution arithmetic for deeplearning에서 추가로 확인 할 수 있습니다.

### 질문들

질문사항1)??? 이해불가

질문사항2) 합성곱 필터는 언제나 정사각형인가요?

답변: 수학적으로는 정사각형일 필요 없지만 아닌 적을 본 적 없습니다.

질문사항3) 합성곱 필터의 크기는 어떻게 정하나요?

답변: 뒤에서 이야기하지만 보통 직관과 경험에 의존합니다.

질문사항4)??? 이해불가

## 5. Convolution Implementation Notes
1. flatten이란 과정을 cnn에서 거치는데 그 이유는 layer를 걸치면서 이미지에 주요 특징만 추출되는데 추출된 주요 특징은 2차원 데이터로 이루어져 있지만, Dense와 같이 분류를 위한 학습 레이어에서는 1차원 데이터로 바꾸어서 학습이 되어야 합니다. 그래서 2차원을 1차원으로 바꾸는 flatten을 합니다!

![2](https://user-images.githubusercontent.com/63699718/119773468-a341cf80-befb-11eb-8850-746c1c8d3c7d.PNG)

2. 그리고 이 과정을 필터가 움직이면서 모든 경우에대해 연산을 하고나면 아래 사진처럼 됩니다.

![3](https://user-images.githubusercontent.com/63699718/119773484-a9d04700-befb-11eb-9c62-52c6bd23b768.PNG)

3. 그 후 아래 사진처럼 row가 만들어집니다.

![4](https://user-images.githubusercontent.com/63699718/119773494-b05ebe80-befb-11eb-871d-b8b4b74bfa50.PNG)

4. 그 후 만들어진 row를 column과 연산(?)을 하고 reshape를 하면 아래처럼 됩니다.

![5](https://user-images.githubusercontent.com/63699718/119773568-c9676f80-befb-11eb-917f-8c4e702b8e72.PNG)

## 6. Increasing the Receptive Field with Dilated Convolutions
receptive field란 output의 특정 weight에 영향을 끼치는 input을 말합니다. 아래 그림에서 3곱하기3 크기에 해당하는 부분이 receptive field라 합니다.

![6](https://user-images.githubusercontent.com/63699718/119774627-670f6e80-befd-11eb-9aa1-347237f81338.PNG)

여기서 receptive field는 conv operation을 쌓음으로써 키울 수 있습니다. 또한 dilated conv를 통해서도 파라미터를 늘리지 않고도 receptive field를 키울 수 있습니다. 여기서 dilated conv란 필터가 지나가면서 연산을 할때 몇 몇 pixel을 건너 뛰는 겁니다. 

![7](https://user-images.githubusercontent.com/63699718/119775396-80fd8100-befe-11eb-881b-10d2a72880ae.PNG)

## 7. Decreasing the Tensor Size with Pooling and 1x1-Convolutions
만약 large image를 input으로 집어넣고 classification처럼 작은 output이 필요하다면 어떻게 해야할까요?
1. 풀링을 통해 줄일 수도 있습니다. 풀링이란 데이터의 차원을 줄이는 방법으로 어떤 영역에서 가장 큰 수만 택하거나 평균을 택해 차원을 줄일 수 있습니다. 

![8](https://user-images.githubusercontent.com/63699718/119776587-14838180-bf00-11eb-923e-b81ff03d6730.PNG)

2. 이미지의 채널을 줄임으로서 목적을 달성할 수도 있습니다.

## 8. LeNet Architecture
아래 사이트 참고

https://arclab.tistory.com/150

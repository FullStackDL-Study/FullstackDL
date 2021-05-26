#Lecture2A: CNNs
##1. Convolutional Filters
합성곱 필터(커널)은 두가지 이유 때문에 도입되었습니다. 첫째는 fully connected network의 경우 학습할때 매우 큰 행렬이 들어가므로 사진 크기를 키워주어야 하고(?) 사진을 키우면 더 많은 파라미터들을 결정해야만 하게 됩니다. 더 많은 weight때문에 성능에 안 좋은 영향을 미칩니다.

![3](https://user-images.githubusercontent.com/63699718/119652853-96709d80-be61-11eb-9e2f-9660b10dc85f.PNG)

둘째는 fully connected network를 사용하면 이미지내에서 중요한 부분과 아닌 부분을 똑같이 취급해서 처리하게 되기 때문입니다. 
필터는 이미지의 특징을 찾아내기 위한 공용 파라미터입니다. 이 커널은 입력 데이터와 합성곱 연산을 수행합니다. 또한 이를 통해 결과 데이터의 차원이 줄어듭니다. 

![58780750-defb7480-8614-11e9-943c-4d44a9d1efc4](https://user-images.githubusercontent.com/63699718/119650876-4b558b00-be5f-11eb-925c-f95bad9693d2.gif)

위 그림은 필터를 적용시키는 걸 보여줍니다. 여기서 패딩과 스트라이드의 개념이 나오는데 이는 아래에 서술합니다.

##2. Filter Stacks and ConvNets
input과 output은 여러 channel를 가질 수 있습니다. 여기서 channel이란 Convolution Layer에 유입되는 입력 데이터에는 한 개 이상의 필터를 말합니다. 1개 필터는 Feature Map의 채널이 됩니다.
Convolution Layer에 n개의 필터가 적용된다면 출력 데이터는 n개의 채널을 갖게 됩니다.

##3. Strides and Padding
스트라이드란 한 번 합성곱 연산한 후 다음 계산 영역을 선택할때 얼마나 이동할지 간격을 정하는 값입니다. 위 사진에서는 stride가 1인 경우 였습니다. 근데 만약 스트라이드가 3쯤이되어서 합성곱하는 부분이 그림을 벗어나게 되면 계산하는데 문제가 생깁니다. 그래서 이를 해결하기위해 padding이 등장합니다. 패딩은 출력 데이터의 차원을 키워주는 방법입니다. 보통 0을 가장자리에 집어넣어 해결합니다. 그럼으로써 스트라이드가 3이여서 생기는 문제도 해결 할 수 있습니다. 

![4](https://user-images.githubusercontent.com/63699718/119655817-f3ba1e00-be64-11eb-9988-6988e12d56c3.PNG)

##4. Filter Math


##5. Convolution Implementation Notes
##6. Increasing the Receptive Field with Dilated Convolutions
##7. Decreasing the Tensor Size with Pooling and 1x1-Convolutions
##8. LeNet Architecture

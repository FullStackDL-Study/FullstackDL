#Lecture2A: CNNs
##1. Convolutional Filters
합성곱 필터(커널)은 두가지 이유 때문에 도입되었습니다. 첫째는 fully connected network의 경우 학습할때 매우 큰 행렬이 들어가므로 사진 크기를 키워주어야 하고 사진을 키우면 더 많은 파라미터들을 결정해야만 하게 됩니다.   
![3](https://user-images.githubusercontent.com/63699718/119652853-96709d80-be61-11eb-9e2f-9660b10dc85f.PNG)
둘째는 fully connected network를 사용하면 이미지내에서 중요한 부분과 아닌 부분을 똑같이 취급해서 처리하게 되기 때문입니다. 
필터는 이미지의 특징을 찾아내기 위한 공용 파라미터입니다. 이 커널은 입력 데이터와 합성곱 연산을 수행합니다. 또한 이를 통해 결과 데이터의 차원이 줄어듭니다. 
![58780750-defb7480-8614-11e9-943c-4d44a9d1efc4](https://user-images.githubusercontent.com/63699718/119650876-4b558b00-be5f-11eb-925c-f95bad9693d2.gif)
위 그림은 필터를 적용시키는 걸 보여줍니다. 여기서 패딩과 스트라이드의 개념이 나오는데 이는 아래에 서술합니다.

##2. Filter Stacks and ConvNets
##3. Strides and Padding

##4. Filter Math
##5. Convolution Implementation Notes
##6. Increasing the Receptive Field with Dilated Convolutions
##7. Decreasing the Tensor Size with Pooling and 1x1-Convolutions
##8. LeNet Architecture

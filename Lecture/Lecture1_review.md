
___
## 1. neural networks
뉴럴넷은 인간의 뉴런에서 영감을 받아 만들어졌다. 뉴런은 만약 충분한 크기의 자극을 받으면 그들은 가지를 따라 다른 뉴런들을 자극시킨다. 
![1](https://user-images.githubusercontent.com/63699718/118811382-e7bfe080-b8e7-11eb-96da-595bf6a08a6f.PNG)
이걸 수학적으로 간단한 함수로 퍼셉트론이라 한다. 인간의 뉴런처럼 충분한 자극을 받으면 자극을 주고 그게 아니라면 자극을 주지 않는다. 여기서 자극을 주는 한계점을 정하는 것이 activation function이다. 
![2](https://user-images.githubusercontent.com/63699718/118811659-39686b00-b8e8-11eb-96f3-61aa65365f06.PNG)
activation function에는 여러개가 있고 그중 몇가지가 sigmoiod function, hyperbolic tangent, rectified linear unit(ReLU) 이다.
![3](https://user-images.githubusercontent.com/63699718/118837696-0c27b700-b900-11eb-9d3c-5e141796f5c7.PNG)
sigmoid의 경우 어떠한 분수가 들어가도 0과1의 두가지 경우만 나온다. ReLU의 경우는 0초과의 값을 넣으면 모두 1이 나오고 아니면 0이 나오는 것이다.
## 2. universality
universal approximation이란 1개의 히든 레이어를 가진 Neural Network를 이용해 어떠한 함수든 근사시킬 수 있다는 이론을 말한다.
## 3. learning problems
머신러닝에는 여러 분류가 있지만 주로 세가지 문제가 있다. 첫번째는 지도학습, 그다음은 비지도 학습 그리고 강화학습이다. 
비지도 학습은 unlabeled 데이터로 학습시키며 주로 데이터의 구조를 알고싶어 한다.
지도 학습은 labeled 데이터를 가지고 학습시킨다. x를 학습시켜 y를 예측시키는 것이 목적이다.
강화학습은 특정한 환경에서 어떤 행동을 하는지 배우게 하는 것이다. 
gpt3, gan, rnn등에대해 간단한 소개가 나온다.
## 4.  empirical Risk minimization / loss functions
linear regression을 예시로 들어보자. 예측을 위해서 데이터에 가장 적합하게 직선을 정해야 한다. 그럼 직선이므로 ax+b에서 a와 b를 구해야한다. a와 b를  squared error를 최소화하는 방법으로 a와b를 구해야 한다. 이걸 일반적으로 loss function이라 하고 우리는 loss function을 최소화 하는 게 목적이다.
## 5. gradient descent
loss function을 최소화하기 위한 weight와 bias를 정해야 한다. 그 방법에는 여러개가 있는데 그중 하나가 SGD(schotastic gradient descent)이다. batch gradient descent라 불리기되 한다. 
![31](https://user-images.githubusercontent.com/63699718/119105331-2bd1f300-ba58-11eb-8e93-d19ad6de56e6.PNG)
## 6. back propagation and automatic differentiation.
우리는 계속 weight를 변경시키며 gradient를 구한다. 그렇다면 어떻게 빠르게 gradient를 계산할 수 있을까? gradient는 미분의 다른 말로 아무리 복잡해도 chain rule을 이용하면 구할 수 있다. 그리고 이계산은 pytorch등이 대신 해줄 것이다. 
## 7. architectural consideration
1)data efficiency
2)optimization landscape, conditioning
## 8. CUDA / Cores of Compute
2013년에 딥러닝이 폭발적인 관심을 받게된 것은 큰 데이터셋과 gpu덕분이다. 모든 NN계산은 matrix인데 gpu가 이걸 매우 잘해낸다. 

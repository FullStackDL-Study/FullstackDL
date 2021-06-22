# RNN
- Sequence problem을 풀기위한 도구   

- 현재의 상태를 입력으로 가지는 순환형 구조

![image](https://user-images.githubusercontent.com/84113554/122913642-a9a95700-d394-11eb-8f41-63cc72da7db2.png)




- Feedforward networks를 쓰지않는 이유   
1. input length의 길이가 변함   
2. timesteps에 선형적인 memory가 요구   
3. overkill

# Vanishing gradients problem
![image](https://user-images.githubusercontent.com/84113554/122914176-55eb3d80-d395-11eb-9fe7-5772766cae7f.png)

back propagation에서 sigmoid & tanh로 인해 충분한 step이후에는 gradient vanishing 문제가 발생

# LSTMs
위와같은 문제를 해결하기 위해 cell state를 채널에 추가
- forget gate : old state의 삭제할 부분을 결정
- input gate : cell state의 update를 결정

# Machine Translation
## LSTMs의 문제점 
1. underfitting => stacked LSTM layers로 해결
2. stacked LSTMs는 training이 어려움 => residual connections (resNet 참고)
3. last timestep에서 너무 많은 정보가 encoding => attention을 통한 해결
4. LSTMs는 이전 context만 고려 => bidirectionality를 통한 미래의 정보를 고려

## GNMT loss function

![image](https://user-images.githubusercontent.com/84113554/122918072-b11f2f00-d399-11eb-8e0f-783cb83d08ae.png)

# CTC loss
![image](https://user-images.githubusercontent.com/84113554/122918302-efb4e980-d399-11eb-954a-b8b2f4a52e0a.png)

input이 달라질 경우 발생하는 문제

- 일반적으로 input과 output의 길이는 다를 수 있다.
- input과 output의 길이의 비는 다를 수 있다.
- X와 Y의 대응하는 elements를 명확히 알 수 없다.

solution : CTC Loss
![image](https://user-images.githubusercontent.com/84113554/122918721-60f49c80-d39a-11eb-8420-b3d4c3d0817d.png)
- 반복되는 요소들을 합침, 엡실론 토큰을 삭제

# Sequence problem에서의 convNet 활용
LSTM은 Long term dependencies, Sequential training required의 한계는 audio modeling을 어렵게 만든다.
![image](https://user-images.githubusercontent.com/84113554/122919877-b5e4e280-d39b-11eb-8f2f-0a7f425eb2ef.png)
- convolution은 sequence data에 사용가능하므로 Receptive field를 크게 가지는 것으로 활용 => dilated convolutions

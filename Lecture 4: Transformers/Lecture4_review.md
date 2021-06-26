# Transfer Learning and Transformers 

## 1. Transfer Learning in Computer Vision 

10K의 라벨링된 이미지가 있고, 새를 분류해본다고 가정해 봅시다. 

ImageNet으로 부터 ResNet-50 과 같은 딥러닝 방식이 잘 작동한다는 것을 알고 있습니다. 하지만 ResNet의 문제점은 모델이 매우 커서 우리가 가지고 있는 10,000개의 이미지 데이터가 오버피팅 될 수가 있습니다.  

이에 대한 한가지 솔루션은, ImageNet으로 학습된 신경망을 사용하되, 새(birds) 데이터에 대해 미세 조정 (fine-tuning) 하는 것입니다. 

이렇게 하면, 기본적인 방식들 보다 높은 성능을 낼 수 있습니다. 

이것이 transfer learning에 대한 기본 개념입니다. 

![image-16](..\img\img-16.png)

전통적인 머신러닝에서는 많은 데이터에 대해 학습하느라, 시간이 매우 오래 거렸지만. Transfer learning을 통해 적은 데이터를 빠르게 학습하여 좋은 성능을 낼 수 있습니다. 

![image-17](..\img\img-17.png)

따라서, ImageNet으로 사전 학습된 모델의 앞부분을 그대로 두고, 끝 부분만 대체하여 transfer-learning을 하게 되면, 

![image-18](..\img\img-18.png)

앞부분은 전과 같은 weight를 가지되, 끝부분만 새로운 weight를 가지게 됩니다. 



Model Zoos 패키지에서 Tensorflow와 Pytorch에 다양한 Deep-Learning 모델을 제공하고 있습니다. 

* [torchvision.models — Torchvision 0.10.0 documentation (pytorch.org)](https://pytorch.org/vision/stable/models.html)

* [models/official at master · tensorflow/models (github.com)](https://github.com/tensorflow/models/tree/master/official)



이를 활용하여 Transfer-learning을 통해 효율적인 딥러닝 모델 학습이 가능합니다. 



## 2. Embeddings and Language Models 

다음은, 자연어 처리(NLP) 에서 입력값 (inputs) 들은 단어들의 sequences 들입니다. 하지만 딥러닝 모델은 vector값 들이 필요합니다. 

그렇다면,어떻게 words를 vector들로 바꿀 수 있을까요? 

방법은 one-hot encoding을 사용하는 것입니다. 

![image-19](..\img\img-19.png)

다음과 같이, 문장에서 단어와 인덱스를 연결하는 단어 사전을 만들고, 벡터들을 0으로 채운뒤에, 각 단어들이 해당하는 index 값으로 대체해주면, 모델이 학습할 수 있는 vector값으로 변환할 수 있습니다. 

예를 들어 the cat is black의 문장은 [8676, 3204, 5281, 2409]의 벡터값으로 변환될 것입니다. 

하지만, 이런 원핫인코딩에는 문제가 있습니다. 

1. 단어 사전의 크기에 따라 Scales이 잘 안될 것이고,
2. 매우 고차원의 희소 벡터들이기 때문에 신경망이 잘 작동되지 않을것이며,
3. 유사 단어들에 대한 처리 문제도 있습니다.
   (e.g "run" is as far away from "running" as from "tertiary" or "poetry.. ??")

그래서, 이 문제를 해결하기위해, V x V matrix을 다음과 같은 V x E embedding matrix으로 변환해볼 수 있습니다.

![image-20](..\img\img-20.png)

좋은 아이디어인데 Embedding  값을 어떻게 찾아야 할까요? 

**첫번째**로는 torch.nn.Embedding 에서 제공하는 Class를 통해, Task에서 학습하는 방식입니다. 

[Embedding — PyTorch 1.9.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

**두번째**로는 언어 모델을 학습하는 것입니다. 텍스트의 큰 corpus를 만들어서  Model 을 pre-training 하는 것입니다. 

![image-21](..\img\img-21.png)

이렇게, Input값으로 Features를 입력하고 Output으로 다음에 올 단어를 예측하는 것입니다. 이러한 모델을 만들기 위해서, 우리는 N-grams 방식으로 dataset을 만들어주어야 하는데요. 아래의 예시는 N = 3 라고 가정했습니다.

![image-22](..\img\img-22.png)

![image-23](..\img\img-23.png)



위와 같이 텍스트로 부터 N 크기의 window를 슬라이딩 시켜서, 마지막 단어를 예측하는 단어 Dataset을 만들 수 있습니다. 

한편 target word의 양 옆을 확인해서, 각 N-gram으로 부터 다수의 samples를 만드는 Skip-grams 방식도 있습니다. 

![image-24](..\img\img-24.png)

학습 속도를 올리기 위해서, multi-class로 하기 보다 Binary로 Task 성격을 바꾸는 방법도 있습니다. 

![image-25](..\img\img-25.png)

input 단어의 다음에 올 단어를 예측하는 것이 아니라, 위의 그림 처럼 두 단어가 이웃한 단어인지 Yes / No 로 예측해보는 것입니다. 이렇게 Task의 성격을 바꾼다면 좀 더 학습시간이 빨라질 수 있습니다. 

![image-26](..\img\img-26.png)

기본적으로 embedding 값을 학습하기 위해서, 타겟 단어 주변에 있는지 없는지를 확인해본다고 해봅시다. 즉, 다시 말해서, target 단어 주변에 input 단어들이 있을 수록 높은 embedding값을 가지고 없을 수록 낮은 embedding값을 가지게 됩니다. 





![image-27](..\img\img-27.png)

단어를 벡터화 해서, 단어들의 관계를 Vector 연산을 통해 계산할 수 있다는 Word2Vec가 2013년에 연구되었습니다.  king 의 vector값에서 man의 vector값을 빼고, woman vector 값을 더했더니, queen의 vector값과 유사했다는 결론인데요. 

![image-28](..\img\img-28.png)



Word2Vec가 연구됨으로써, 많은 단어들의 관계를 파악할 수 있었습니다. 





## 3. NLP's ImageNet moment: ELMO and ULMFit on datasets like SQuAD, SNLI, and GLUE

2013 ~ 2014년도에 Word2Vec와 GloVe embedding기법들이 유명하지기 시작했습니다.  많은 task들에 대해 10% 정도 정확도를 향상시켰지만, 문제점은 이 기법들이 shallow 하다는 것이었습니다. 오직 첫번째 레이어만 pre-training된 모델의 이점을 갖게 되어 우리가 학습시킬 Wikipedia의 정보를 보고, 다른 레이어에는 정보가 소실됩니다. 그러면, pre-training model을 좀 더 깊게 구성하여 문법을 학습시키는 (e.g. rule in "to rule" vs "a rule")  것이 가능할까요? 

이를 처음으로 시도한 것이 2018년도에 연구된 ELMO (Embeddings from Language Model) 입니다. 

![image-29](..\img\img-29.png)



예를 들어, Bank Account(은행 계좌) 와 River Bank(강둑) 에서의 Bank가 전혀 다른 의미를 가지는데, 이전의 Word2Vec이나 GloVe 등으로 표현된 임베딩 벡터들은, 이를 제대로 반영하지 못했습니다. 하지만 ELMO는 LSTM을 양방향으로 쌓아올려, 문맥을 반영한 워드 임베딩을 구현했습니다. 

[09) 엘모(Embeddings from Language Model, ELMo) - 딥 러닝을 이용한 자연어 처리 입문 (wikidocs.net)](https://wikidocs.net/33930)

이 방법을 통해, 여러 Task에서 성능 향상을 이끌어냈다고 합니다. 

![image-30](..\img\img-30.png)



한편, SQuAD는 100K의 질문-답변 쌍으로 구성되어있는 텍스트 데이터입니다. 

![image-31](..\img\img-31.png)

오른쪽 부분에 있는 In meteorology, precipitation ~~ 은 source text이고, 아래에 What causes precipitation to fall? 에 대한 답변은 source text에 있는 gravity입니다.  이런 식의 질문 답변 쌍으로 구성되어 있습니다. 

또 다른, 자연어처리 데이터셋으로  Natural Language Inference:  SNLI가 있는데요,

![image-32](..\img\img-32.png)

이 데이터셋은 570,000 쌍의 텍스트관에 관계를 나타낸 데이터셋 입니다. 

한편 GLUE (General Language Inderstanding Evaluation)라는 평가 방법을 통해, 자연어 처리 성능을 검증하게 되는데요. 

![image-33](..\img\img-33.png)

하나가 아니라, 여러가지 (9개) Task를 수행하고 그값을 취합하여 최종 점수를 얻는 방식으로, Pre-training 모델의 성능을 측정하는데 많이 사용된다고 합니다. 

또 다른 imagenet moment model은 ULMFit 이라는, 2018년도에 개발된 모델이 있는데요. 이 모델은 언어 모델 기반 문서 Classifier 라고 할 수 있습니다. 

![image-34](..\img\img-34.png)



이런 시도들을 통해 많은 모델들이 개발되었고, Tensorflow나 Pytorch에서 활용할 수 있습니다. 

[models/official at master · tensorflow/models (github.com)](https://github.com/tensorflow/models/tree/master/official)

[Model Zoo - PyTorch Deep Learning Code and Models](https://modelzoo.co/framework/pytorch)

(pytorch는 각 모델을 라이브러리로 설치해야되는 것 (pip install) 같네요.)

##  

## 4. Rise of Transformers



![image-35](..\img\img-35.png)

NLP관련 모델을 살펴보시면, BERT (Bi-directional Encoder Representation from Transformer) , NHNet (transformer 기반 sequence to sequence 모델) 등등 Transformer라고 적혀있는 것을 볼 수 있습니다. 

그럼 Transformer가 무엇일까요? 이 모델은 2017년도의 Attention is all you need라는 논문에서 시작됩니다. 

![image-36](..\img\img-36.png)

이 모델은 encoder - decoder 아키텍쳐로 구성되어있고, 어떤 lstm이나 rnn 계열의 신경망 layer가 없습니다. 오직 attention과 fully-connected layers로만 구성되었습니다. 이 모델은 Translation dataset의 SOTA가 되었습니다. 

메커니즘이 어떻게 돌아가는지 좀 더 깊게 봅시다. 

![image-37](..\img\img-37.png)

Encoder만 보면, (Masked) Self-attention, Positional encoding, Layer normalization 으로 구성되어있습니다. 

![image-38](..\img\img-38.png)

기본적인 self-attention은 sequence of tensor를 입력값으로 받고, sequence of tensor를 출력합니다. 각 출력값은 input sequnce들의 가중합입니다.  

여기서, w는 학습된 가중치가 아니며, x_i 와 x_j를 연산하여, 합계가 1이 되도록 변환한 값입니다. 

![image-39](..\img\img-39.png)

그림으로 나타내면 위와 같이, input 값으로 The Cat is yawning의 vector값, output 값으로 각 vector들의 가중합의 결과가 출력됩니다. 

그렇다면, 학습된 가중치가 아니었던  **w**를 학습된 가중치로 바꾸게 되면 어떻게 될까요? 

![image-40](..\img\img-40.png)

입력된 vector_xi는 3가지 방식으로 사용됩니다. 

1. 첫번째로, **자신의** output y_i **(query)** 에 해당하는 attention weight를 계산하기 위해. 다른 모든 vector들과 비교합니다. 
2. 두번째로,  ouput y_j **(key)**에 해당하는 attention weight w_ij 를 계산하기 위해, 다른 모든 vector들과 비교합니다. 
3. 다른 벡터들을 합쳐서 결과로 attention weighted sum (가중합) **(value)** 를 도출합니다. 

![image-41](..\img\img-41.png)



## 5. Attention in Detail: (Masked) Self-Attention, Positional Encoding, and Layer Normalization 

![image-42](..\img\img-42.png)

조금 더 들어가서, Multiple "heads" of attention은 서로 다른 W_q, W_k, W_v 행렬을 동시에 학습하는 것을 의미합니다.  하지만 구현해보는것은 하나의 행렬로 해볼 것입니다. head가 여러개라면 그 수만큼 거질 것이기 때문입니다. 

![image-43](..\img\img-43.png)

정리하자면, Transfomer는 위와 같이, Input vector들이 self-attention layer를 거치고, layer normalization을 거쳐서 MultiLayerPerceptron의 Denselayer를 거친 다음, output vector를 출력합니다. 

그렇다면, 여기서 Layer Normalization이란 무엇일까요? 

![image-44](..\img\img-44.png)

Layer Normlization은 input vector가 각 dimension 마다, uniform mean과 std를 가질 때, 가장 학습이 잘 수행 된다는 발견에서 시작되었습니다.  당신은 기본적으로 당신의 input data가 어떤 방향으론, variance가 매우 작고, 어떤 방향으로는 variance가 매우 크길 원하지 않을 것 입니다.  그래서  input data를 scaling하고, weight initialization 해서 입력 데이터를 보정할 수 있습니다. 하지만, 처음 weight 초기화가 아무리 좋더라도, 입력 값이 network를 지나가면, 평균과 분산이 계속해서 바뀔 텐데요. 그래서 layer normalization은 이를 reset 하여 layer간에 mean과 std가 uniform을 유지하도록 해줍니다. 

지금까지 설명한 내용은 다음과 같습니다. 

* query, key, value weight에 대해 학습했습니다. 
* Multiple heads에 대해 배웠습니다. 
* 하지만 **sequence의 순서**가 계산 결과에 영향을 주지 않습니다. 



그래서, 나온 개념이 Positional Embedding 입니다. 

![image-45](..\img\img-45.png)

각 단어는 위와 같이 dense vector로 embedding이 될 것 입니다. 하지만 word embedding은 단어의 순서를 고려하지 않기 때문에 여기에 position embedding을 추가합니다. position embedding은 단어가 무엇인지는 고려하지 않고, 오직 단어의 순서만 고려합니다. 이 두가지를 결합하여 transfomer block에 입력하고, 그 결과로 output sequence를 출력하게 됩니다. 



마지막으로, Transformer는 모든 input들을 한번에 넣기 때문에, 아래와 같이 다음 단어가 무엇이 오는지 예측하는 과제에서 문제가 생깁니다. 

![image-46](..\img\img-46.png)

그래서 input data에 mask를 추가하여 input이 transformer blocks를 통해 attention weight를 계산한 후,  future blocking mask를 weight에 적용합니다. 그러면 attention을 할 때, 이전에 온 단어들 만 attention을 하여 이 문제를 해결할 수 있습니다. 

![image-47](..\img\img-47.png)



## 6. Transformers Variants: BERT, GPT/GPT-2/GPT-3, DistillBERT, T5, etc

attention은 이해하기 어려우니까, Reading article을 통해 좀 더이해해 보세요. 

2017년도에 Attentnion is all you need 논문이 발표된 후, encoder만 있거나 decoder만 있는 다양한 모델들이 개발되었는데요. 최신 모델은 encoder decoder 로 돌아왔긴 합니다. 

![image-48](..\img\img-48.png)

GPT나 GPT2 , GPT3는 Generative Pre-trained Transformer의 약자로서, ELMO나 ULMFIT 처럼, sequnce에서 다음 단어를 예측하도록 학습하였습니다. ELMO나 ULMFIT는 LSTM을 사용했지만, GPT는 LSTM을 transformer로 대체한 것입니다

![image-49](..\img\img-49.png)

그리고 skip-gram이 아닌 n-gram처럼, 이전의 단어에만 영향을 받기 때문에, 전에 이야기했단 Masked Self-Attention 기법을 사용했습니다. 그래서 오직 preceding word에만 attention할 수 있게 되었습니다. 

![image-50](..\img\img-50.png)

GPT에도 다양한 변형 모델들이 있는데요, parameter 수에 따라 Large Small등 다양하게 불립니다. 

[Talk to Transformer – InferKit](https://app.inferkit.com/demo)

여기 웹사이트에서 1.5billion parameter를 가진 extra large GPT 모델을 사용해볼 수 있습니다. 말되는것 같진 않는데 어쨋든 잘 출력하네요. 



다음은 BERT (Bidirectional Encoder Representations from Transformer) 모델인데요.  GPT가 not bi-directional 이었다면, BERT는 bi-directional 기능을 추가한 것입니다.  또, mask를 추가하여 다음 단어를 예측했다면, BERT는 bi-directional이기 때문에 no-masking 입니다. 

![image-51](..\img\img-51.png)

![image-52](..\img\img-52.png)

BERT는 340M parameter를 가지고, 24개의 transformer block, embedding size는 1024개, 16개의 attention heads 를 가집니다.  



다음은 2020년 2월에 개발된 T5 모델입니다. 

![image-53](..\img\img-53.png)

기본적으로,  이 모델은 사람들이 서로 다른 transformer 모델을 가지고 했던 모든 paper들을 평가하여 Hyper-parameter tuning을 하였습니다.  input과 output 모두 text strings으로 되어있고 , C4 corpus로 학습하여 GLUE, SuperGLUE, SQuAD의 SOTA가 되었습니다. 그래서 이번 연구에 흥미가 있다면 t5 모델을 써보는 것을 추천드립니다. 



## 7. GPT3 Demos

[100+ GPT-3 Examples, Demos, Apps, Showcase, and NLP Use-cases | GPT-3 Demo (gpt3demo.com)](https://gpt3demo.com/)



## 8. 참고할 추천 자료

[huggingface/transformers: 🤗Transformers: State-of-the-art Natural Language Processing for Pytorch, TensorFlow, and JAX. (github.com)](https://github.com/huggingface/transformers)

[Transformer Explained | Papers With Code](https://paperswithcode.com/method/transformer)

[The latest in Machine Learning | Papers With Code](https://paperswithcode.com/)

[sebastianruder/NLP-progress: Repository to track the progress in Natural Language Processing (NLP), including the datasets and the current state-of-the-art for the most common NLP tasks. (github.com)](https://github.com/sebastianruder/NLP-progress)


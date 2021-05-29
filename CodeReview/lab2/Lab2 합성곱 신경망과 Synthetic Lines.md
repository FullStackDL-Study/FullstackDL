

# Lab 2: 합성곱 신경망과 Synthetic Lines

## 이번 Lab의 목표

손으로 쓴 이미지를 텍스트로 번역하는 작업을 해볼 것입니다. 이번 lab에서는 다음을 진행할 예정입니다. 

* 단순 합성곱 네트워크를 사용하여 EMNIST 문자를 인식합니다. 
* EMNIST의 synthetic 데이터셋 line을 구축합니다.



## 시작하기 전에 환경설정을 완료해주세요! 

진행하기 전에 [Lab Setup]([fsdl-text-recognizer-2021-labs/readme.md at main · full-stack-deep-learning/fsdl-text-recognizer-2021-labs (github.com)](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/setup/readme.md)) 을 완료해주세요! 

그리고나서, `fsdl-text-recognizer-2021-labs` 레퍼지토리안에서, 마지막 변경사항을 pull 해주세요. 그러면 정확한 디렉토리에 들어가게 됩니다. 



## Colab Ver.

```python
# FSDL Spring 2021 Setup
!git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
%cd fsdl-text-recognizer-2021-labs

!pip install pytorch_lightning
%env PYTHONPATH=.:$PYTHONPATH
```

<img src=".\img\image-1.png" height = "200px">

git clone을 하면 다음과 같이 colab의 content 폴더 내에 git-repository가 clone되어 추가됩니다. 



```python
%cd lab2
>>> /content/fsdl-text-recognizer-2021-labs/lab2
```

작업경로를 lab2 폴더로 이동해주세요. 



### EMIST 소개

MNIST는 Mini-NIST를 나타냅니다. NIST는 National Institute of Standards and Technology 이며, 1980년대에 손으로 쓴 숫자와 문자 데이터셋을 구축하였습니다. 

MNIST는 오직 숫자만 포함하기 때문에 작습니다. 

EMNIST는 문자도 포함하며 널리 알려진 MNIST 형식으로 제공되는 원본 데이터셋을 리패키징한 것입니다.  아래의 링크에서 관련된 논문을 확인할 수 있습니다. 

https://www.paperswithcode.com/paper/emnist-an-extension-of-mnist-to-handwritten

`notebooks/01-look-at-emnist.ipynb`에서 데이터를 살펴보겠습니다. 

(지금 `lab2`에 `notebooks`라는 새로운 디렉토리가 생겼습니다. 노트북에서 우리의 모델을 학습시키지 않지만, 데이터를 탐색하는데 노트북을 사용할 수 있습니다, 그리고 아마 우리의 모델 학습의 결과를 프리젠팅할 때도 사용할 수 있을 것입니다.)

[fsdl-text-recognizer-2021-labs/01-look-at-emnist.ipynb at main · full-stack-deep-learning/fsdl-text-recognizer-2021-labs (github.com)](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab2/notebooks/01-look-at-emnist.ipynb)

#### Looking at EMNIST data (노트북 설명)

```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from importlib.util import find_spec
if find_spec("text_recognizer") is None:
    import sys
    sys.path.append('..')

from text_recognizer.data.emnist import EMNIST
```

* EMNIST 모듈을 import 합니다. 

<img src=".\img\image-2.png" height = "200px">

* 다음 폴더의 emnist.py 모듈에 있는 EMNIST class를 불러오는 의미입니다. 



```python
data = EMNIST()
data.prepare_data()
data.setup()
print(data)
>>> 
EMNIST Dataset
Num classes: 83
Mapping: ['<B>', '<S>', '<E>', '<P>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']
Dims: (1, 28, 28)
Train/val/test sizes: 260212, 65054, 53988
Batch x stats: (torch.Size([128, 1, 28, 28]), torch.float32, tensor(0.), tensor(0.1656), tensor(0.3257), tensor(1.))
Batch y stats: (torch.Size([128]), torch.int64, tensor(4), tensor(65))
```

* EMNIST class를 불러와서 prepare_data() 함수와 setup() 함수로 데이터를 준비하여 data를 print 하면 위와 같이 출력됩니다. 

* class갯수와 , class가 무엇을 뜻하는지 (대문자, 소문자, 숫자, 특수기호 등) 와 , Dimension (차원),  Train/val/test 사이즈 등이 출력됩니다. 
* <B> , <S> , <E> , <P> 는 다음 강의에 설명할 special token입니다. 주로 rnn 계열의 순환신경망을 위해 사용되는 token입니다. 



```python
x, y = next(iter(data.test_dataloader()))
print(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())
print(y.shape, y.dtype, y.min(), y.max())

>>> 
torch.Size([128, 1, 28, 28]) torch.float32 tensor(0.) tensor(0.1656) tensor(0.3257) tensor(1.)
torch.Size([128]) torch.int64 tensor(4) tensor(65)
```

* test_data마다 로드하여 배치마다 x,y를 출력하면, 위와 같은 torch 정보가 출력됩니다. 



```python
fig = plt.figure(figsize=(9, 9))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    rand_i = np.random.randint(len(data.data_test))
    image, label = data.data_test[rand_i]
    ax.imshow(image.reshape(28, 28), cmap='gray')
    ax.set_title(data.mapping[label])
```

<img src=".\img\image-3.png" height = "500px">

* 각 이미지를 출력해보았습니다. class가 무엇인지와 어떤 그림인지 출력되었습니다. 

#### CNN 모델 학습

```python
import pytorch_lightning as pl
from text_recognizer.models import CNN
from text_recognizer.lit_models import BaseLitModel

model = CNN(data_config=data.config())
lit_model = BaseLitModel(model=model)
trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(lit_model, datamodule=data)
>>> 
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name      | Type     | Params
---------------------------------------
0 | model     | CNN      | 1.7 M 
1 | train_acc | Accuracy | 0     
2 | val_acc   | Accuracy | 0     
3 | test_acc  | Accuracy | 0     
---------------------------------------
1.7 M     Trainable params
0         Non-trainable params
1.7 M     Total params
6.616     Total estimated model params size (MB)
```

<img src=".\img\image-4.png">

* 5 epoch을 학습시켜보겠습니다. GPU를 활성화시켜야, 학습이 5분내로 끝난다는점 주의 해주세요. 
* pytorch_lightning에서 cnn 모델을 import 하여 lit_model을 학습시켜줍니다. 

```python
x, y = next(iter(data.test_dataloader()))

logits = model(x)  # (B, C)
print(logits.shape)

preds = logits.argmax(-1)

print(y, preds)

>>> 
torch.Size([128, 83])
tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4]) tensor([28,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         4, 28,  4,  4,  4,  4,  4,  4, 28,  4,  4,  4,  4,  4,  4,  4,  4,  4,
        28, 28,  4,  4, 28,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 28,  4,  4,
         4,  4,  4,  4,  4,  4,  4, 28,  4, 28,  4, 28, 28,  4,  4,  4,  4,  4,
         4, 28,  4,  4, 28,  4,  4, 28,  4,  4, 28,  4,  4,  4,  4,  4, 28,  4,
         4,  4,  4,  4,  4, 28,  4,  4,  4, 28,  4,  4,  4,  4,  4,  4,  4,  4,
         4, 28,  4,  4, 28,  4, 17, 28,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
        28, 28])
```

* model이 predict 한 값과 실제 값을 확인해 볼 수 있습니다. 얼추 비슷하게 4로 잘 예측한 것 같습니다. 

```python
fig = plt.figure(figsize=(9, 9))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    rand_i = np.random.randint(len(data.data_test))
    image, label = data.data_test[rand_i]

    image_for_model = image.unsqueeze(0)  # (1, 1, 28, 28)
    logits = model(image_for_model)  # (1, C)
    pred_ind = logits.argmax(-1)  # (1, )
    pred_label = data.mapping[pred_ind]

    ax.imshow(image.reshape(28, 28), cmap='gray')
    ax.set_title(f'Correct: {data.mapping[label]}, Pred: {pred_label}')
```

<img src=".\img\image-5.png" height = "500px">

* 이제 모델이 예측한 것과 실제 값을 시각화 하여 확인해보겠습니다. 
* 랜덤으로 9개 출력해 보았는데 전부 잘 예측한 것 같습니다. 



### MNIST를 인식하기 위한 합성곱 네트워크 사용하기

위의 과정은 terminal 환경에서도 할 수 있습니다. 

Lab1에서 MNIST 숫자 dataset으로 MLP 모델을 학습시켜보았습니다. 

이제 같은 목적으로 CNN을 학습시켜볼 수 있습니다. 

colab에서는 ! 느낌표를 앞에다가 적으면 terminal에서 입력한 것과 동일하게 실행됩니다. 

```python
!pip install wandb
!python3 training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```

* max_epoch은 5, gpus는 1, data_class 는 MNIST로 모듈을 run할 수 있습니다. 노트북 보다 훨씬더 빠르게 실행이 됩니다. 

<img src=".\img\image-6.png" height = "200px">

* 실행하면 `lab2/text_recognizer/models` 폴더에 `cnn.py` 모듈이 새로 생성되었을 겁니다. 
* `cnn.py` 모듈을 확인하면, 네트워크 구조가 어떻게 설계되었는지 확인해볼 수 있습니다.
* 저의 경우, 'test_acc' : 0.987 정도 나왔네요 



### EMNIST로 합성곱 네트워크 사용하기

좀 더 큰 EMNIST dataset으로 같은 작업을 해볼 수 있습니다. 

```python
!python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1
```

한 epoch당 학습하는데 2분정도 걸립니다. (이번 lab에서 한 epoch만 수행해보는 이유입니다 :) ) 

* 저의 경우 'test_acc' : 0.806 정도 나왔네요 ! 



### overfitting 의도하기 

빠른 실험을 위해 데이터셋을 subsample 할 수 있고, 모델이 데이터를 나타낼 수 있을 정도로 견고함을 확인하는 것은 중요합니다. 

###### - Subsampling : 원본 데이터의 부분집합을 선택하여 데이터 크기를 줄여주는 방법

(더 자세한 사항은 Training & Debugging 강의에서 설명하겠습니다)

이는 `--overfit_batches = 0.01` 을 전달함으로써 가능합니다. 또한 구체적인 배치 갯수 대신에 int `>1` 을 전달할 수 있습니다.  자세한 사항은 아래 링크를 확인하세요. 

https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html?highlight=make%20model%20overfit#make-model-overfit-on-subset-of-data

```python
!python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=50 --gpus=1 --overfit_batches=2
```

* 저의 경우 'test_acc' 가 1.0으로 나왔네요! 
* 학습속도도 훨씬 더 빠릅니다. 
* pytorch-lightning doc에서 확인해보니, class 당 data의 작은 부분을 (class당 sample 2개) 가지고 와서, 모델에 overfitting을 시켜보는 디버깅 기법이라고 합니다. 만약에 overfitting이 되지 않는다면, 만든 모델은 large dataset에 작동하지 않는 다는 것을 뜻한다고하네요. 



### 학습 속도 높이기 

우리의 GPU가 지속적으로 높은 활용도를 유지하도록 하는 한가지 방법은 별도의 worker 프로세스를 추가하여 데이터 사전 처리를 수행하는 것입니다. 

```python
!python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1 --num_workers=4
```

* num_workers = 4로 두었더니, 훨씬 더 학습 속도가 빨라진 것을 확인할 수 있습니다.

* test_acc 도 동일하게 0.806 정도 나왔네요. 



### EMNIST  의 synthetic 데이터셋 Lines 만들기

* `notebooks/02-look-at-emnist-lines.ipynb`를 확인하세요

[fsdl-text-recognizer-2021-labs/02-look-at-emnist-lines.ipynb at main · full-stack-deep-learning/fsdl-text-recognizer-2021-labs (github.com)](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab2/notebooks/02-look-at-emnist-lines.ipynb)

이 실습은 자기가 직접 손글씨를 써보고, 잘 인식되는지 확인해보는 실습입니다.! 



(수정 중)



### Homework

cnn.py를 resnet과 유사하게 바꿔서 모델 성능 높여보기 

(수정 중)

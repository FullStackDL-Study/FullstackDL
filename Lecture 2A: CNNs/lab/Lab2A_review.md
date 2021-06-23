

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
* `<B> , <S> , <E> , <P> `는 다음 강의에 설명할 special token입니다. 주로 rnn 계열의 순환신경망을 위해 사용되는 token입니다. 



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

* 이 실습의 목적은 EMNIST 글자들이 합쳐진 synthetic dataset을 만들어 한줄에 출력해보는 것입니다. 

* 옵션으로 약간의 랜덤 overlap을 줄 수도 있습니다. 



```python
%matplotlib inline
import matplotlib.pyplot as plt
import nltk
import numpy as np

%load_ext autoreload
%autoreload 2

from importlib.util import find_spec
if find_spec("text_recognizer") is None:
    import sys
    sys.path.append('..')

from text_recognizer.data.emnist_lines import EMNISTLines, construct_image_from_string, get_samples_by_char
from text_recognizer.data.sentence_generator import SentenceGenerator
```

* EMNISTLines라는 새로운 데이터를 import 해줍니다. 
* 기본적인 아이디어는 현실의 손글씨를 인식해서 텍스트를 생성해내는 것입니다. 



```python
sentence_generator = SentenceGenerator()
for _ in range(4):
    print(sentence_generator.generate(max_length=16))
    
>>>
name anyway
accept
in school
the Democratic
```

* 처음으로 현실에서 generate 할 문장을 출력합니다.  그리고 나서 가짜 손글씨 line을 출력할 것입니다.
* 이 `sentence_generator` 는 `brown corpus` 를 사용했습니다. 
  * `brown corpus` 는 다양한 장르의 구조화된 영어 텍스트 샘플 말뭉치입입니다. 
* sentence의 `max_length` 는 16으로 설정하여 랜덤으로 문장을 출력했습니다. 저는 name anyway accept in school the Democratic 이 나왔네요 

 

```python
import argparse
args = argparse.Namespace(max_length=16, max_overlap=0)
dataset = EMNISTLines(args)
dataset.prepare_data()
dataset.setup()
print(dataset)
print('Mapping:', dataset.mapping)

>>> 
EMNIST Lines Dataset
Min overlap: 0
Max overlap: 0
Num classes: 83
Dims: (1, 28, 448)
Output dims: (16, 1)
Train/val/test sizes: 10000, 2000, 2000
Batch x stats: (torch.Size([128, 1, 28, 448]), torch.float32, tensor(0.), tensor(0.0777), tensor(0.2379), tensor(1.))
Batch y stats: (torch.Size([128, 16]), torch.int64, tensor(3), tensor(66))

Mapping: ['<B>', '<S>', '<E>', '<P>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A
```

* 그리고 생성된 문장을 생성해서 EMNIST Lines Dataset을 만들어줍니다. 

* `max_overlap` 은 문장간 겹치는 것이 없도록 설정하는 파라미터입니다.



```python
def convert_y_label_to_string(y, dataset=dataset):
    return ''.join([dataset.mapping[i] for i in y])

y_example = dataset.data_train[0][1]
print(y_example, y_example.shape)
convert_y_label_to_string(y_example)

>>> 
[48 59 66 45 54 57 42 44  3  3  3  3  3  3  3  3] (16,)
it force<P><P><P><P><P><P><P><P>
```

* 여기에서 볼 수 있는 `<P>` 는 Padding을 위한 special token입니다.  `sentence_generator` 의 max_length가 16이기 때문에 단어 이외의 나머지 공간을 `<p>` 로 채우는 것입니다. 



```python
num_samples_to_plot = 9

for i in range(num_samples_to_plot):
    plt.figure(figsize=(20, 20))
    x, y = dataset.data_train[i]
    sentence = convert_y_label_to_string(y) 
    print(sentence)
    plt.title(sentence)
    plt.imshow(x.squeeze(), cmap='gray')
    
>>> 
it force<P><P><P><P><P><P><P><P>
at this years<P><P><P>
the<P><P><P><P><P><P><P><P><P><P><P><P><P>
unsupportable<P><P><P>
that way<P><P><P><P><P><P><P><P>
59<P><P><P><P><P><P><P><P><P><P><P><P><P><P>
world<P><P><P><P><P><P><P><P><P><P><P>
followed<P><P><P><P><P><P><P><P>
as<P><P><P><P><P><P><P><P><P><P><P><P><P><P>
```

<img src=".\img\image-7.png" >

(이하 이미지생략)

생성된 EMNIST Lines가 나빠보이진 않네요. 



다음은 조금 더 어렵게 한 줄에 글자의 최대 수를 확장해보고, 글자 간의 overlap 값도 랜덤 값으로 전달해보겠습니다. 

```python
args = argparse.Namespace(max_length=34, max_overlap=0.33)
dataset = EMNISTLines(args)
dataset.prepare_data()
dataset.setup()
print(dataset)

>>> 
EMNIST Lines Dataset
Min overlap: 0
Max overlap: 0.33
Num classes: 83
Dims: (1, 28, 952)
Output dims: (34, 1)
Train/val/test sizes: 10000, 2000, 2000
Batch x stats: (torch.Size([128, 1, 28, 952]), torch.float32, tensor(0.), tensor(0.0805), tensor(0.2416), tensor(1.))
Batch y stats: (torch.Size([128, 34]), torch.int64, tensor(3), tensor(66))
```

* 마찬가지로 EMNISTLines의 dataset을 만들어주고, `max_length` 를 34까지 늘리고, `max_overlap`을 0.33 값으로 줍니다. 



```python
num_samples_to_plot = 9

for i in range(num_samples_to_plot):
    plt.figure(figsize=(20, 20))
    x, y = dataset.data_train[i]
    sentence = convert_y_label_to_string(y) 
    print(sentence)
    plt.title(sentence)
    plt.imshow(x.squeeze(), cmap='gray')
    
>>> 
phase of Newark will<P><P><P><P><P><P><P><P><P><P><P><P><P><P>
the fire for washing<P><P><P><P><P><P><P><P><P><P><P><P><P><P>
go much further than<P><P><P><P><P><P><P><P><P><P><P><P><P><P>
up Dont<P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P>
resolutely at<P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P>
abrupt end<P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P>
high ground<P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P><P>
voyage The sickness was<P><P><P><P><P><P><P><P><P><P><P>
I said Its a kindness to<P><P><P><P><P><P><P><P><P><P>
```

<img src=".\img\image-8.png" >

* 조금 더 생성된 단어들이 문장스러워 졌습니다. 하지만 말이 되는것 같진 않네요. 



### Homework

cnn.py를 resnet과 유사하게 바꿔서 모델 성능 높여보는 것이 숙제입니다. 

Residual block은 아래와 같습니다. 

<img src=".\img\image-9.png" height='400'>

아래의 사항을 시도 해보세요. 

* `BatchNorm` 과 같은 ResNet의 secret sauce를 추가해보세요 . 공식 ResNet PyTorch 실행을 참고하면 아이디어를 확인해볼 수 있습니다. 

  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

* `MaxPool2D`를 제거해 보세요. 아마 대신에 strided convolution을 사용해야 할 것입니다. 

* 더 빠른 연산을 수행할 다른 인자들을 command-line에 추가해보세요 

* 추가할 좋은 인자는 입력을 실행할 ConvBlocks의 수에 대한 것입니다. 



밑에 보시기 전에 **자기가 직접 수정**해보시길 바랍니다:) 아래 부터는 저의 삽질 과정이 기록되어 있습니다. 











먼저 cnn.py의 forward 함수 부분을 보겠습니다. 

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

* 전에 학습을 수행했을 때는, test_acc가 0.806 정도 나왔습니다. 

* 여기있는 forward 함수부분을 residual block 처럼 아래와 같이 수정해보겠습니다. 



```python
 def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x_residual = x 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x += x_residual 
        x = F.relu(x)
        return x
```

오류가 뜨네요. 이렇게 하면 안되나봅니다.  



```python
RuntimeError                              Traceback (most recent call last)
<ipython-input-27-4e604aa010a2> in <module>()
      6 lit_model = BaseLitModel(model=model)
      7 trainer = pl.Trainer(gpus=1, max_epochs=5)
----> 8 trainer.fit(lit_model, datamodule=data)

14 frames
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py in nll_loss(input, target, weight, size_average, ignore_index, reduce, reduction)
   2388         ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
   2389     elif dim == 4:
-> 2390         ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
   2391     else:
   2392         # dim == 3 or dim > 4

RuntimeError: 1only batches of spatial targets supported (3D tensors) but got targets of size: : [128]
```

저는 이런 RuntimeError가 나오네요. target size는 [128] 인데..  torch의 shape가 안맞아서 계산이 안되나 봅니다. 어떻게 하면 좋을까요? 

살짝 pytorch의 공식 resnet.py를 보고 디버깅을 시작해봐야겠네요 .



**Q1. Whats the difference between nn.relu() vs F.relu()** 

여기에서 쓰인 CNN.py는 relu를 F.relu()함수를 사용했고, ResNet.py에선 nn.relu()를 사용했습니다. 

검색해보니, `nn.ReLU()` module은 `nn.Sequential` 모듈과 같은 것을 모델에 추가할 수 있지만, `nn.functional.relu` 는 `forward` method에 추가할 수 있다고 하네요. 코딩 스타일 문제이지 큰 차이 없는 것 같습니다.  



최대한 원본 `foward` 를 건드리지 않고, skip connection 아이디어만 추가하여 아래와 같이 `foward` method를 수정해보니 코드가 돌아가네요. 과연 성능이 향상되었을 까요? 



```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE

        identity = x 
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```



수정된 CNN.py를 저장해주고, 아래 코드를 재실행하여 모델을 재학습시켜줍니다. 

```python 
import pytorch_lightning as pl
from text_recognizer.models import CNN
from text_recognizer.lit_models import BaseLitModel

model = CNN(data_config=data.config())
lit_model = BaseLitModel(model=model)
trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(lit_model, datamodule=data)
```

<img src=".\img\image-10.png" >

코드를 수정하기 전에는 val_loss : 0.568, val_acc : 0.788 이었는데, skip connection 아이디어를 추가하니 val_loss : 0.605, val_acc : 0.777 로 loss는 증가하고 acc는 떨어졌네요...계속 진행해보겠습니다. 



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

```python 
import pytorch_lightning as pl
from text_recognizer.models import CNN
from text_recognizer.lit_models import BaseLitModel

model = CNN(data_config=data.config())
lit_model = BaseLitModel(model=model)
trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(lit_model, datamodule=data)
```

<img src=".\img\image-11.png" >

이번에는 틀린 것도 나오네요. 첫번째 그림은 2인데 Z 로 예측했습니다. 이건 사람이 봐도 잘 모를것 같아요. 

이제 터미널에서 `run_experiment.py` 모듈을 실행해보겠습니다. 

```python
!python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1 --num_workers=4
```

전에는 test_acc : 0.806 이었는데, 어떻게 될까요? 

```python
>>> DATALOADER:0 TEST RESULTS
    {'test_acc': 0.7758020162582397}
```

아 성능이 올라가진 않았네요. 제대로된 Residual Block으로 모델을 구성한 뒤, 학습을 해야 신경망을 깊게 구성하여 성능을 올릴 수 있나봅니다. 


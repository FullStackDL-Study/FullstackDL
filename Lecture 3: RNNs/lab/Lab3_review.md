# Lab 3 : CTC loss로 CNN + LSTM을 사용하여 한 줄의 텍스트 인식하기 



## Local vscode.ver

**Tip** : cmd terminal에서도 VSCode 가상환경 실행 안될 때, (conda 명령어를 실행할 수 없습니다 오류)

1. anaconda prompt에서는 잘 되는지 확인 : conda activate fsdl-text-recognizer-2021

2. vscode의 cmd terminal 실행  

3. C:\Users\diane\anaconda3 \Scripts\ 경로로 이동 

   `cd C:\Users\diane\anaconda3\Scripts`

4. activate.bat 실행  해서 (base) 가 뜨는지 확인
5. conda activate fsdl-text-recognizer-2021 로 가상환경 활성화 



## Colab.ver (recommend)

* 런타임 > 런타임유형변경 > 하드웨어가속기 > GPU 실행 

```python
!python --version
>>> Python 3.7.10 
```

* 현재 작업중인 python version 은 3.7 version임 



**cf)** colab python version 3.6으로 맞춰주는 방법  (강의와 동일한 환경)

* 이번 lab에서는 굳이 할 필요 없음 

```python
%%bash

MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
```

```python
import sys
_ = (sys.path.append("/usr/local/lib/python3.6/site-packages"))
```

```python
!python --version
>>> Python 3.6.5 :: Anaconda, Inc.
```

* 해당 노트북에서만 python 3.6 version으로 실행됌.  

  

```python
# FSDL Spring 2021 Setup
!git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
%cd fsdl-text-recognizer-2021-labs
!pip install boltons wandb pytorch_lightning==1.1.4 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
%env PYTHONPATH=.:$PYTHONPATH
%cd lab3
```

```python
# 현재 작업경로 확인 
import os
os.getcwd()
>>> '/content/fsdl-text-recognizer-2021-labs/lab3'
```



## 목표 

* 한 이미지에 다수의 글자를 읽을 수 있는 모델인 `LineCNNSimple` 을 소개합니다. 

* 이 모델이 더욱 효율적으로 되도록 `LineCNN`을 만듭니다. 

* `LitModelCTC`으로 CTC loss를 소개합니다. 
* `LineCNNLSTM`으로 CNN의 꼭대기 층에 있는 LSTM 층을 소개합니다. 



## New files

```python
├── text_recognizer
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── emnist_essentials.json
│   │   ├── emnist_lines.py
│   │   ├── emnist.py
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   ├── sentence_generator.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   ├── ctc.py              <-- NEW
│   │   ├── __init__.py
│   │   ├── metrics.py          <-- NEW
│   │   └── util.py             <-- NEW
│   ├── models
│   │   ├── cnn.py
│   │   ├── __init__.py
│   │   ├── line_cnn_lstm.py    <-- NEW
│   │   ├── line_cnn.py         <-- NEW
│   │   ├── line_cnn_simple.py  <-- NEW
│   │   └── mlp.py
│   └── util.py
```

### LineCNNSipmle :  다수의 글자를 한번에 읽기

이제, 한 글자가 아닌 한 줄의 dataset을 가지고 있습니다.  우리는 우리으 컨볼루젼 신경망을 여기에 적용할 수 있습니다. 

어떻게 data가 생겼는지 리마인드 하기 위해, 다시 한번 `notebooks/02-look-at-emnist-lines.ipynb`를 봅시다. 

우리가 시도할 첫번째 모델은 `CNN` 류의 simple wrapper이며, 데이터를 각 사각형의 슬라이스를 통하여 입력 이미지를 sequence로 적용합니다: `LineCNNSimple`

다음으로, 이 코드를 봅시다. 

우리는 이 모델을 다음의 명령어로 학습할 수 있습니다. 

```python 
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=28
```

(ps. 저는 ValueError: Probabilities in preds must sum up to 1 across the `C` dimension. 가 뜨네요....)

> **trouble shooting** 
>
> lab3 > text_recognizer > lit_models > base.py 
>
> Accuracy class 수정: dim = -1 => dim = 1 
>
> ![image-12](.\img\img-12.png)
>
> [Lab 3 - base.py Acccuracy.update() Error: · Issue #27 · full-stack-deep-learning/fsdl-text-recognizer-2021-labs · GitHub](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/issues/27)
>
> Lab3에서는 Lab1, 2와는 달리, 글자들의 sequence들을 예측하기 위한 3차원의 예측치를 갖게 된다고 합니다. [ shape : (128,83,32) = (batch_size, num_classes, len_seq) ] 
>
> 그래서 softmax함수를 통과할 때, dim = 1 로 수정 해야 정확한 num_classes가 입력 되기 때문에 오류가 해결된다고 하네요.

이렇게 하면, 90% 의 정확도를 얻습니다.  (ps. 저는 0.82 정도 얻었습니다)



### Loss Function

우리는 여전히 `BaseLitModel`을 디폴트 손실 함수로 `cross_entropy` 를 사용하고 있다는 점을 주목합시다. 이 함수에 대해 [PyTorch docs]([torch.nn.functional — PyTorch 1.9.0 documentation](https://pytorch.org/docs/stable/nn.functional.html#cross-entropy)) 을 읽어보면, 여러개의 레이블을 사용할 수 있다는 것을 알 수 있습니다. -- 이는 "K-차원의" 손실이라고 부릅니다. 



### window_stride 바꾸기

더 나아가서, 이번엔 window_stride를 바꿔봅시다. 그리하여 우리는 windows를 겹치기위해 sampling할 것입니다.

* window_stride : 28 => 20

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=20
```

* 이전에 실행했던 코드에서는 window_width가 28, window_stride가 28 이었기 때문에 windows가 아래와 같이 겹치지 않았습니다. 

![image-14](.\img\img-14.png)

* 하지만 이번 code에서는 window_with = 28, window_stride = 20 이기 때문에 아래와 같이 window가 겹치게 됩니다. 

![image-13](.\img\img-13.png)

* Oops! 실행해보니까 아래와 같은 오류 메세지가 나오네요.

```python
>>> ValueError: Expected target size (128, 44), got torch.Size([128, 32])
```

* `--limit_output_length` 라는 하나의 추가적인 flag를 추가해야겠습니다.  새로운 stride 때문에, 우리의 모델이 기대한 ground truth (정답 값) 보다 다른 길이의 sequence를 출력하기 때문입니다.  (우리가 CTC loss를 사용하게 되면 문제가 되지 않을 것 입니다.)

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

* 이번에는 그렇게 높은 정확도를 가지진 않네요 ( 최대 <60% 정도입니다 ) (ps. 저는 0.4 정도가 나왔습니다. ) 왜냐하면, 이 데이터셋은 실제로 글자들간에 겹치지 않았는데, 우리 모델은 겹친다고 예상했기 때문이죠. 



### Changing overlap

우리의 새로운 `window_stride`를 매칭시키기 위해, 우리의 합성(synthetic) 데이터셋을 0.25 정도 overlap 시키겠습니다. 

* min_overlap : 0 => 0.25

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

이렇게 하면 정확도가 80% 정도로 나오네요 (ps. 저는 0.53 정도가 나옵니다.)



### Variable-length overlap

우리는 만약 우리의 모델의 `window_stride` 가 우리 데이터의 overlap된 것과 매치가 된다면, 성공적으로 학습할 것으로 보고 있습니다. 

실제 손글씨는 다양한 스타일을 가지고 있습니다. : 어떤 사람들은 글자들 끼리 매우 가깝게 쓰기도하고, 어떤 사람들은 멀리 떨어져 쓰기도 합니다. 그리고 글자들의 너비도 다릅니다.  우리의 synthetic 데이터들을 좀 더 이와 유사하게 만들어주기 위해, 우리는 `--min_overlap=0 --max_overlap = 0.33` 으로 설정해주겠습니다. 

* min_overlap : 0.25 -> 0 
* max)overlap : 0 -> 0.33 

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

예상했듯이, 우리의 모델은 이 non-uniform overlap 양을 잘 다루지 못하는 것 같습니다. 최고 accuracy는 ~60% 정도 얻었습니다. (ps. 저는 0.49정도 나왔습니다 )



## LineCNN: 좀 더 효율적으로 만들기

한 줄 씩 CNN모델로 읽어들이는 단순한 구현은 위에서 잘 해보았습니다. 하지만 `window_stride`가 `window_with`보다 작다면 꽤 비효율적입니다. 왜냐하면 CNN 모델을 통해서 각 입력값을 window에 각각 보내기 때문입니다.  

우리는 이 작업을 `LineCNN` 이라고 하는 완전-합성곱 모델 (fully-convolutional model)을 통해 향상시킬 수 있습니다. 

계속 진행해서 모델 코드를 한번 보겠습니다. 

* text_recognizer > models > line_cnn.py  : class LineCNN 

  ```python
  from typing import Any, Dict, Union, Tuple
  import argparse
  import math
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  
  
  # Common type hints
  Param2D = Union[int, Tuple[int, int]]
  
  CONV_DIM = 32
  FC_DIM = 512
  WINDOW_WIDTH = 16
  WINDOW_STRIDE = 8
  
  
  class ConvBlock(nn.Module):
      """
      Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
      """
  
      def __init__(
          self,
          input_channels: int,
          output_channels: int,
          kernel_size: Param2D = 3,
          stride: Param2D = 1,
          padding: Param2D = 1,
      ) -> None:
          super().__init__()
          self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
          self.relu = nn.ReLU()
  
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          """
          Parameters
          ----------
          x
              of dimensions (B, C, H, W)
  
          Returns
          -------
          torch.Tensor
              of dimensions (B, C, H, W)
          """
          c = self.conv(x)
          r = self.relu(c)
          return r
  
  
  class LineCNN(nn.Module):
      """
      Model that uses a simple CNN to process an image of a line of characters with a window, outputs a sequence of logits
      """
  
      def __init__(
          self,
          data_config: Dict[str, Any],
          args: argparse.Namespace = None,
      ) -> None:
          super().__init__()
          self.data_config = data_config
          self.args = vars(args) if args is not None else {}
          self.num_classes = len(data_config["mapping"])
          self.output_length = data_config["output_dims"][0]
  
          _C, H, _W = data_config["input_dims"]
          conv_dim = self.args.get("conv_dim", CONV_DIM)
          fc_dim = self.args.get("fc_dim", FC_DIM)
          self.WW = self.args.get("window_width", WINDOW_WIDTH)
          self.WS = self.args.get("window_stride", WINDOW_STRIDE)
          self.limit_output_length = self.args.get("limit_output_length", False)
  
          # Input is (1, H, W)
          self.convs = nn.Sequential(
              ConvBlock(1, conv_dim),
              ConvBlock(conv_dim, conv_dim),
              ConvBlock(conv_dim, conv_dim, stride=2),
              ConvBlock(conv_dim, conv_dim),
              ConvBlock(conv_dim, conv_dim * 2, stride=2),
              ConvBlock(conv_dim * 2, conv_dim * 2),
              ConvBlock(conv_dim * 2, conv_dim * 4, stride=2),
              ConvBlock(conv_dim * 4, conv_dim * 4),
              ConvBlock(
                  conv_dim * 4, fc_dim, kernel_size=(H // 8, self.WW // 8), stride=(H // 8, self.WS // 8), padding=0
              ),
          )
          self.fc1 = nn.Linear(fc_dim, fc_dim)
          self.dropout = nn.Dropout(0.2)
          self.fc2 = nn.Linear(fc_dim, self.num_classes)
  
          self._init_weights()
  
      def _init_weights(self):
          """
          Initialize weights in a better way than default.
          See https://github.com/pytorch/pytorch/issues/18182
          """
          for m in self.modules():
              if type(m) in {
                  nn.Conv2d,
                  nn.Conv3d,
                  nn.ConvTranspose2d,
                  nn.ConvTranspose3d,
                  nn.Linear,
              }:
                  nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                  if m.bias is not None:
                      _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(  # pylint: disable=protected-access
                          m.weight.data
                      )
                      bound = 1 / math.sqrt(fan_out)
                      nn.init.normal_(m.bias, -bound, bound)
  
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          """
          Parameters
          ----------
          x
              (B, 1, H, W) input image
  
          Returns
          -------
          torch.Tensor
              (B, C, S) logits, where S is the length of the sequence and C is the number of classes
              S can be computed from W and self.window_width
              C is self.num_classes
          """
          _B, _C, _H, _W = x.shape
          x = self.convs(x)  # (B, FC_DIM, 1, Sx)
          x = x.squeeze(2).permute(0, 2, 1)  # (B, S, FC_DIM)
          x = F.relu(self.fc1(x))  # -> (B, S, FC_DIM)
          x = self.dropout(x)
          x = self.fc2(x)  # (B, S, C)
          x = x.permute(0, 2, 1)  # -> (B, C, S)
          if self.limit_output_length:
              x = x[:, :, : self.output_length]
          return x
  
      @staticmethod
      def add_to_argparse(parser):
          parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
          parser.add_argument("--fc_dim", type=int, default=FC_DIM)
          parser.add_argument(
              "--window_width",
              type=int,
              default=WINDOW_WIDTH,
              help="Width of the window that will slide over the input image.",
          )
          parser.add_argument(
              "--window_stride",
              type=int,
              default=WINDOW_STRIDE,
              help="Stride of the window that will slide over the input image.",
          )
          parser.add_argument("--limit_output_length", action="store_true", default=False)
          return parser
  ```

또한 우리는 모델을 고정된-overlap 데이터셋으로 학습할 수 있습니다. 

* min_overlap : 0 -> 0.25
* max_overlap : 0 -> 0.25 
* model_class : LineCNNSimple -> LineCNN 

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNN --window_width=28 --window_stride=20 --limit_output_length
```

이전 모델과 비슷하게 수행하네요. (ps. 저는 0.419 정도 나왔습니다.)



## CTC Loss

그리고 이제, 우리의 문제를 해결할 solution인 CTC loss에 대해 봅시다. 

이를 사용하기 위해서 `CTCLitModel`을 소개드리겠습니다. `--loss = ctc`라고 세팅하면 사용할 수 있습니다. 

code를 보기위해 몇 가지를 언급드리겠습니다. 

* Start, Blank, Padding tokens 
* `torch.nn.CTCLoss` function
* `CharacterErrorRate`
* `.greedy_decode()`
* text_recognizer > lit_models > ctc.py  의 CTCLitModel 

```python
import argparse
import itertools
import torch

from .base import BaseLitModel
from .metrics import CharacterErrorRate
from .util import first_element


def compute_input_lengths(padded_sequences: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    padded_sequences
        (N, S) tensor where elements that equal 0 correspond to padding

    Returns
    -------
    torch.Tensor
        (N,) tensor where each element corresponds to the non-padded length of each sequence

    Examples
    --------
    >>> X = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 0, 0], [1, 2, 3, 0, 5]])
    >>> compute_input_lengths(X)
    tensor([2, 3, 5])
    """
    lengths = torch.arange(padded_sequences.shape[1]).type_as(padded_sequences)
    return ((padded_sequences > 0) * lengths).argmax(1) + 1


class CTCLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        inverse_mapping = {val: ind for ind, val in enumerate(self.model.data_config["mapping"])}
        start_index = inverse_mapping["<S>"]
        self.blank_index = inverse_mapping["<B>"]
        end_index = inverse_mapping["<E>"]
        self.padding_index = inverse_mapping["<P>"]

        self.loss_fn = torch.nn.CTCLoss(zero_infinity=True)
        # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

        ignore_tokens = [start_index, end_index, self.padding_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)

        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = first_element(y, self.padding_index).type_as(y)
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S  # All are max sequence length
        target_lengths = first_element(y, self.padding_index).type_as(y)  # Length is up to first padding token
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("val_loss", loss, prog_bar=True)

        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.val_acc(decoded, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(decoded, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.test_acc(decoded, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_cer(decoded, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

    def greedy_decode(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Greedily decode sequences, collapsing repeated tokens, and removing the CTC blank token.

        See the "Inference" sections of https://distill.pub/2017/ctc/

        Using groupby inspired by https://github.com/nanoporetech/fast-ctc-decode/blob/master/tests/benchmark.py#L8

        Parameters
        ----------
        logprobs
            (B, C, S) log probabilities
        max_length
            max length of a sequence

        Returns
        -------
        torch.Tensor
            (B, S) class indices
        """
        B = logprobs.shape[0]
        argmax = logprobs.argmax(1)
        decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.padding_index
        for i in range(B):
            seq = [b for b, _g in itertools.groupby(argmax[i].tolist()) if b != self.blank_index][:max_length]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded
```

우리의 현재 모델에서 CTC loss를 추가해 봅시다. 

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNN --window_width=28 --window_stride=20 --loss=ctc
```

* Oooops! 실행하니까 저는 또 아래와 같은 RuntimeError가 나오네요 ㅠ 

```python
>>> RuntimeError: "host_softmax" not implemented for 'Int'
```

> **trouble shooting**
>
> lab3 > text_recognizer > lit_models > base.py 
>
> preds => preds.float()로 수정 
>
> ![image-15](.\img\img-15.png)
>
> https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/issues/28
>
> softmax로 들어갈 입력값의 type이 int로 들어가게 되나 봅니다. 

오류를 해결하고 코드를 실행하면 CER이 10 epochs 안에 ~18% 미만으로 나오게 됩니다. 

* (ps. 저는 test_acc : 0, test_cer : 0.0730 으로 나왔습니다. ...? 왜 이렇게 적게 나올까요)
* (ps. cer (Character Error Rate) 모델 예측과 실제 글자의 오차 )

최고로, 우리는 이제 variable-overlap data를 다룰 수 있게 되었습니다.

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNN --window_width=28 --window_stride=18 --loss=ctc
```

이 코드는 ~15% CER이 나옵니다.

* (ps. 저는 test_acc : 0, test_cer : 0.0796 으로 나왔습니다 . 이전과 비슷하네요)



## LSTM 추가하기

마지막으로, 우리는 `LineCNN`의 제일 윗 층에 LSTM을 추가하여 더 나은 성능 향상을 확인할 수 있습니다. 

이 모델은 `LineCNNLSTM` 입니다.  보기 위해 시간을 좀 들여봅시다. 

* text_recognizer > models > line_cnn_lstm.py   LineCNNLSTM class

```python
from typing import Any, Dict
import argparse
import torch
import torch.nn as nn

from .line_cnn import LineCNN

LSTM_DIM = 512
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2


class LineCNNLSTM(nn.Module):
    """Process the line through a CNN and process the resulting sequence through LSTM layers."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}

        num_classes = len(data_config["mapping"])
        lstm_dim = self.args.get("lstm_dim", LSTM_DIM)
        lstm_layers = self.args.get("lstm_layers", LSTM_LAYERS)
        lstm_dropout = self.args.get("lstm_dropout", LSTM_DROPOUT)

        self.line_cnn = LineCNN(data_config=data_config, args=args)
        # LineCNN outputs (B, C, S) log probs, with C == num_classes

        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and CHAR_WIDTH
            C is num_classes
        """
        x = self.line_cnn(x)  # -> (B, C, S)
        B, _C, S = x.shape
        x = x.permute(2, 0, 1)  # -> (S, B, C)

        x, _ = self.lstm(x)  # -> (S, B, 2 * H) where H is lstm_dim

        # Sum up both directions of the LSTM:
        x = x.view(S, B, 2, -1).sum(dim=2)  # -> (S, B, H)

        x = self.fc(x)  # -> (S, B, C)

        return x.permute(1, 2, 0)  # -> (B, C, S)

    @staticmethod
    def add_to_argparse(parser):
        LineCNN.add_to_argparse(parser)
        parser.add_argument("--lstm_dim", type=int, default=LSTM_DIM)
        parser.add_argument("--lstm_layers", type=int, default=LSTM_LAYERS)
        parser.add_argument("--lstm_dropout", type=float, default=LSTM_DROPOUT)
        return parser
```

우리는 아래의 코드를 실행하여 학습할 수 있습니다. 

* min_overlap = 0.25 -> 0 
* max_overlap = 0.25 -> 0.33
* window_stride = 20 -> 18 

```python
!python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNLSTM --window_width=28 --window_stride=18 --loss=ctc
```

(ps. 저는 test_acc : 0 , test_cer : 0.0601 정도로 나왔습니다. 비슷하네요 )



## Homework 

두 부분 입니다: 

### Experiments

CNN의 hyperparameters (`window_width`, `window_stride`, `conv_dim`, `fc_dim`)  그리고 / 또는 LSTM 의 hyperparameters (`lstm_dim`, `lstm_layers`) 가지고 놀아보세요

더 좋아지면, residual connections을 사용하여 `LineCNN`을 수정해보세요. 그리고 Lab 2에서 했던 것 처럼 다른 CNN tricks를 사용하거나 몇가지 방식으로 아키텍쳐를 바꿔보세요.  

`LineCNNLSTM`를 자유롭게 수정해보세요. LSTM에 미쳐보세요! (..?)

### CTCLitModel

당신의 말로, 어떻게 `CharacterErrorRate` 평가지표 (metric) 와 `greedy_decode` method가 작동하는지 설명해보세요. 


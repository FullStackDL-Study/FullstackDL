## 디렉토리 구조

```sh
(fsdl-text-recognizer-2021) ➜  lab1 git:(main) ✗ tree -I "logs|admin|wandb|__pycache__"
.
├── readme.md
├── text_recognizer
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   └── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── mlp.py
│   └── util.py
└── training
    ├── __init__.py
    └── run_experiment.py
```
# text_recognizer/data
데이터를 다루기 위한 모든 파일들이 이 디렉토리에 있습니다.

[base_data_module.py](#base_data_module.py)

[util.py](#util.py)

# text_recognizer/lit_models
라이트닝 모듈의 기저 클래스를 정의합니다.

[base.py](#base.py)

- [PyTorch Lightning](https://baeseongsu.github.io/posts/pytorch-lightning-introduction/) 관련 


# base_data_module.py  
```python
"""Base DataModule class."""
from pathlib import Path
from typing import Dict
import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from text_recognizer import util


def load_and_print_info(data_module_class: type) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)

def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename


```
필요한 라이브러리들을 import하고 load_and_print_info()를 정의합니다. 

add_to_argparse()는 아래에서 staticmethod로 클래스 내부에 정의되지만 self를 argument로 받지 않아 클래스 객체 instantiation 없이 호출할 수 있습니다. (추가설명 필요)

_ download_raw_dataset()함수를 정의, 앞의 언더바는 이 함수가 내부 사용을 권장한다는 의미이다. 지정된 위치에 디렉토리를 만들고, 다운로드 후 처리 과정을 거칩니다. 디렉토리를 만들 때, 부모 디렉토리가 없으면 그것도 생성하고, 이미 있는 디렉토리라면 오류 없이 다음으로 넘어갑니다.


```python
BATCH_SIZE = 128
NUM_WORKERS = 0


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {} # To Dictionary
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        # Make sure to set the variables below in subclasses
        self.dims = None
        self.output_dims = None
        self.mapping = None
```

pytorch_lightning.LightningDataModule을 상속하는 BaseDataModule 클래스를 정의합니다. 이후 사용하는 데이터셋은 모두 BaseDataModule을 상속하므로, 전부 LightningDataModule의
subclass라고 할 수 있습니다. ()

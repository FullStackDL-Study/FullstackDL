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

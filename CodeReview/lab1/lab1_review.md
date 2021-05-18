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

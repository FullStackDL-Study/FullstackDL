# Lab 4: Recognize synthetic sequences with Transformers

## Colab.ver

* 런타임 > 런타임유형변경 > 하드웨어가속기 > GPU 실행 

```python
# FSDL Spring 2021 Setup
!git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
%cd fsdl-text-recognizer-2021-labs
!pip install boltons wandb pytorch_lightning==1.1.4 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
%env PYTHONPATH=.:$PYTHONPATH
% cd lab4
```

## Training
```python
python training/run_experiment.py --max_epochs=40 --gpus=1 --num_workers=16 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNTransformer --window_width=20 --window_stride=12 --loss=transformer

```

## Result
```
  | Name                                                       | Type                    | Params
---------------------------------------------------------------------------------------------------------
0   | model                                                      | LineCNNTransformer      | 3.8 M 
1   | model.line_cnn                                             | LineCNN                 | 1.1 M 
2   | model.line_cnn.convs                                       | Sequential              | 698 K 
3   | model.line_cnn.convs.0                                     | ConvBlock               | 320   
4   | model.line_cnn.convs.0.conv                                | Conv2d                  | 320   
5   | model.line_cnn.convs.0.relu                                | ReLU                    | 0     
6   | model.line_cnn.convs.1                                     | ConvBlock               | 9.2 K 
7   | model.line_cnn.convs.1.conv                                | Conv2d                  | 9.2 K 
8   | model.line_cnn.convs.1.relu                                | ReLU                    | 0     
9   | model.line_cnn.convs.2                                     | ConvBlock               | 9.2 K 
10  | model.line_cnn.convs.2.conv                                | Conv2d                  | 9.2 K 
11  | model.line_cnn.convs.2.relu                                | ReLU                    | 0     
12  | model.line_cnn.convs.3                                     | ConvBlock               | 9.2 K 
13  | model.line_cnn.convs.3.conv                                | Conv2d                  | 9.2 K 
14  | model.line_cnn.convs.3.relu                                | ReLU                    | 0     
15  | model.line_cnn.convs.4                                     | ConvBlock               | 18.5 K
16  | model.line_cnn.convs.4.conv                                | Conv2d                  | 18.5 K
17  | model.line_cnn.convs.4.relu                                | ReLU                    | 0     
18  | model.line_cnn.convs.5                                     | ConvBlock               | 36.9 K
19  | model.line_cnn.convs.5.conv                                | Conv2d                  | 36.9 K
20  | model.line_cnn.convs.5.relu                                | ReLU                    | 0     
21  | model.line_cnn.convs.6                                     | ConvBlock               | 73.9 K
22  | model.line_cnn.convs.6.conv                                | Conv2d                  | 73.9 K
23  | model.line_cnn.convs.6.relu                                | ReLU                    | 0     
24  | model.line_cnn.convs.7                                     | ConvBlock               | 147 K 
25  | model.line_cnn.convs.7.conv                                | Conv2d                  | 147 K 
26  | model.line_cnn.convs.7.relu                                | ReLU                    | 0     
27  | model.line_cnn.convs.8                                     | ConvBlock               | 393 K 
28  | model.line_cnn.convs.8.conv                                | Conv2d                  | 393 K 
29  | model.line_cnn.convs.8.relu                                | ReLU                    | 0     
30  | model.line_cnn.fc1                                         | Linear                  | 262 K 
31  | model.line_cnn.dropout                                     | Dropout                 | 0     
32  | model.line_cnn.fc2                                         | Linear                  | 131 K 
33  | model.embedding                                            | Embedding               | 21.2 K
34  | model.fc                                                   | Linear                  | 21.3 K
35  | model.pos_encoder                                          | PositionalEncoding      | 0     
36  | model.pos_encoder.dropout                                  | Dropout                 | 0     
37  | model.transformer_decoder                                  | TransformerDecoder      | 2.6 M 
38  | model.transformer_decoder.layers                           | ModuleList              | 2.6 M 
39  | model.transformer_decoder.layers.0                         | TransformerDecoderLayer | 659 K 
40  | model.transformer_decoder.layers.0.self_attn               | MultiheadAttention      | 263 K 
41  | model.transformer_decoder.layers.0.self_attn.out_proj      | _LinearWithBias         | 65.8 K
42  | model.transformer_decoder.layers.0.multihead_attn          | MultiheadAttention      | 263 K 
43  | model.transformer_decoder.layers.0.multihead_attn.out_proj | _LinearWithBias         | 65.8 K
44  | model.transformer_decoder.layers.0.linear1                 | Linear                  | 65.8 K
45  | model.transformer_decoder.layers.0.dropout                 | Dropout                 | 0     
46  | model.transformer_decoder.layers.0.linear2                 | Linear                  | 65.8 K
47  | model.transformer_decoder.layers.0.norm1                   | LayerNorm               | 512   
48  | model.transformer_decoder.layers.0.norm2                   | LayerNorm               | 512   
49  | model.transformer_decoder.layers.0.norm3                   | LayerNorm               | 512   
50  | model.transformer_decoder.layers.0.dropout1                | Dropout                 | 0     
51  | model.transformer_decoder.layers.0.dropout2                | Dropout                 | 0     
52  | model.transformer_decoder.layers.0.dropout3                | Dropout                 | 0     
53  | model.transformer_decoder.layers.1                         | TransformerDecoderLayer | 659 K 
54  | model.transformer_decoder.layers.1.self_attn               | MultiheadAttention      | 263 K 
55  | model.transformer_decoder.layers.1.self_attn.out_proj      | _LinearWithBias         | 65.8 K
56  | model.transformer_decoder.layers.1.multihead_attn          | MultiheadAttention      | 263 K 
57  | model.transformer_decoder.layers.1.multihead_attn.out_proj | _LinearWithBias         | 65.8 K
58  | model.transformer_decoder.layers.1.linear1                 | Linear                  | 65.8 K
59  | model.transformer_decoder.layers.1.dropout                 | Dropout                 | 0     
60  | model.transformer_decoder.layers.1.linear2                 | Linear                  | 65.8 K
61  | model.transformer_decoder.layers.1.norm1                   | LayerNorm               | 512   
62  | model.transformer_decoder.layers.1.norm2                   | LayerNorm               | 512   
63  | model.transformer_decoder.layers.1.norm3                   | LayerNorm               | 512   
64  | model.transformer_decoder.layers.1.dropout1                | Dropout                 | 0     
65  | model.transformer_decoder.layers.1.dropout2                | Dropout                 | 0     
66  | model.transformer_decoder.layers.1.dropout3                | Dropout                 | 0     
67  | model.transformer_decoder.layers.2                         | TransformerDecoderLayer | 659 K 
68  | model.transformer_decoder.layers.2.self_attn               | MultiheadAttention      | 263 K 
69  | model.transformer_decoder.layers.2.self_attn.out_proj      | _LinearWithBias         | 65.8 K
70  | model.transformer_decoder.layers.2.multihead_attn          | MultiheadAttention      | 263 K 
71  | model.transformer_decoder.layers.2.multihead_attn.out_proj | _LinearWithBias         | 65.8 K
72  | model.transformer_decoder.layers.2.linear1                 | Linear                  | 65.8 K
73  | model.transformer_decoder.layers.2.dropout                 | Dropout                 | 0     
74  | model.transformer_decoder.layers.2.linear2                 | Linear                  | 65.8 K
75  | model.transformer_decoder.layers.2.norm1                   | LayerNorm               | 512   
76  | model.transformer_decoder.layers.2.norm2                   | LayerNorm               | 512   
77  | model.transformer_decoder.layers.2.norm3                   | LayerNorm               | 512   
78  | model.transformer_decoder.layers.2.dropout1                | Dropout                 | 0     
79  | model.transformer_decoder.layers.2.dropout2                | Dropout                 | 0     
80  | model.transformer_decoder.layers.2.dropout3                | Dropout                 | 0     
81  | model.transformer_decoder.layers.3                         | TransformerDecoderLayer | 659 K 
82  | model.transformer_decoder.layers.3.self_attn               | MultiheadAttention      | 263 K 
83  | model.transformer_decoder.layers.3.self_attn.out_proj      | _LinearWithBias         | 65.8 K
84  | model.transformer_decoder.layers.3.multihead_attn          | MultiheadAttention      | 263 K 
85  | model.transformer_decoder.layers.3.multihead_attn.out_proj | _LinearWithBias         | 65.8 K
86  | model.transformer_decoder.layers.3.linear1                 | Linear                  | 65.8 K
87  | model.transformer_decoder.layers.3.dropout                 | Dropout                 | 0     
88  | model.transformer_decoder.layers.3.linear2                 | Linear                  | 65.8 K
89  | model.transformer_decoder.layers.3.norm1                   | LayerNorm               | 512   
90  | model.transformer_decoder.layers.3.norm2                   | LayerNorm               | 512   
91  | model.transformer_decoder.layers.3.norm3                   | LayerNorm               | 512   
92  | model.transformer_decoder.layers.3.dropout1                | Dropout                 | 0     
93  | model.transformer_decoder.layers.3.dropout2                | Dropout                 | 0     
94  | model.transformer_decoder.layers.3.dropout3                | Dropout                 | 0     
95  | train_acc                                                  | Accuracy                | 0     
96  | val_acc                                                    | Accuracy                | 0     
97  | test_acc                                                   | Accuracy                | 0     
98  | loss_fn                                                    | CrossEntropyLoss        | 0     
99  | val_cer                                                    | CharacterErrorRate      | 0     
100 | test_cer                                                   | CharacterErrorRate      | 0     
---------------------------------------------------------------------------------------------------------
3.8 M     Trainable params
0         Non-trainable params
3.8 M     Total params
Epoch 0:  83% 79/95 [00:28<00:05,  2.80it/s, loss=3.03, v_num=0, val_loss=4.99, val_cer=0.993]
Validating: 0it [00:00, ?it/s]
Epoch 0:  85% 81/95 [00:29<00:05,  2.70it/s, loss=3.03, v_num=0, val_loss=4.99, val_cer=0.993]
Validating:  12% 2/16 [00:02<00:19,  1.42s/it]
Epoch 0:  87% 83/95 [00:31<00:04,  2.66it/s, loss=3.03, v_num=0, val_loss=4.99, val_cer=0.993]
...
...
Epoch 39: 100% 95/95 [00:42<00:00,  2.26it/s, loss=0.0999, v_num=0, val_loss=0.151, val_cer=0.583]
Epoch 39: 100% 95/95 [00:42<00:00,  2.26it/s, loss=0.0999, v_num=0, val_loss=0.151, val_cer=0.583]
EMNISTLinesDataset loading data from HDF5...
Testing: 100% 16/16 [00:10<00:00,  1.50it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_cer': tensor(0.5835, device='cuda:0')}
--------------------------------------------------------------------------------
```
  

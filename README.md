# BiCVM++
This repository is for my undergraduate thesis *'BiCVM++: Representing the internal semantics of sentence'*. I implement BiCVM++ with PyTorch here.

## Prerequisites
**PyTorch**

This implementation is based on PyTorch 1.0.1 ([https://pytorch.org/](https://pytorch.org/)), I am not sure whether it works on PyTorch 0.4.x. If you find something wrong when you run it on PyTorch 0.4.x, feel free to contact me.

**CUDA**

CUDA is suggested ([https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)) for fast inference. You need to run it on GPU for this version of BiCVM++.

## Quick Start
The current release has been tested on Ubuntu 16.04.6 LTS.

### **Clone the repository**
```
sh git clone https://github.com/Coldog2333/BiCVM-plus-plus
```

### **Download datasets and models**
```sh
cd .
wget http://coldog.cn/nlp/bicvmpp/data.zip
unzip data.zip
wget http://coldog.cn/nlp/bicvmpp/models.zip
unzip models.zip
```
### **Pre-train BiCVM++**
```
python3 train.py [[option][value]]...
```

**Options**
+ **--data:** The dataset that you want to train on. Valid Values: [tiny, 1K, 17K, 170K, full], default: tiny
+ **--act:** Activation utilized in the pipeline.
+ **--noise:** The number of noisy samples. default: 25
+ **--model:** The name of model you want to save as. default value depends on waht dataset you use.
+ **--bid:** Use bidirectional LSTM? [T/F]. default: False
+ **--gpuID:** The No. of the GPU you want to use. default: 0
+ **--mode:** Do you want to use MemoryFriendlyLoader or CorpusLoader? Valid values: [MemoryFriendly/Effective], default: MemoryFriendly
+ **-h, --help:** Get help.

**Example**
```
python3 train.py --data full --act tanh --noise 25 --mode MemoryFriendly --bid F
```

### **Fine-tune on Sentiment Analysis**
```
python3 SA_finetune.py [[option][value]]...
```


**Options**
+ **--act:** activation utilized in Pipeline. Valid values:[tanh, ptf, penalized tanh]. default: penalized tanh
+ **--freeze:** Do you want to freeze the parameters of CVM? [T/F]. default: False
+ **--pretrain:** Do you want to load pretrain weights? [T/F]. default: True
+ **--bid:** Use bidirectional LSTM? [T/F]. default: False
+ **--mode:** Do you want to use MemoryFriendlyLoader or CorpusLoader? [MemoryFriendly/Effective]
+ **--model:** the name of model you want to save as. default: temp 
+ **--gpuID:** the No. of the GPU you want to use. default: 1
+ **--task:** special for the baseline Just for Sentiment Analysis. Valid values: ['Just4SA', 'just', 'Just']
+ **-h, --help:** Get help.

**Example**

+ finetune full BiCVM++
```
python3 SA_finetune.py --model SA --act tanh --freeze F --pretrain T --gpuID 1 --mode Effective --bid F
```

+ finetune the baseline, Just for Sentiment Analysis
```
python3 SA_finetune.py --model SA_just --act ptf --gpuID 1 --mode Effective --task just
```

### **Evaluate**
```
python3 SA_test.py [[option][value]]...
```

**Options**
+ **--act:** activation utilized in Pipeline. Valid values: [tanh, ptf, penalized tanh]. default: penalized tanh
+ **--bid:** Use bidirectional LSTM? [T/F]. default: False
+ **--model:** the name of model you want to use. default: temp 
+ **--gpuID:** the No. of the GPU you want to use. default: 1
+ **--task:** special for the baseline Just for Sentiment Analysis. Valid values: ['Just4SA', 'just', 'Just']
+ **-h, --help:** Get help.

**Example**

Evaluate full BiCVM++
```
python3 SA_test.py --model SA --act tanh --gpuID 1 --bid F
```


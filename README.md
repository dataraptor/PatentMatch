# PatentMatch
A Transformer based model for Patent Word matching

This project is focused on developing a machine learning model to match phrases in U.S. patents based on their semantic similarity within a specific technical domain context. The goal is to assist patent attorneys and examiners in retrieving relevant documents and connecting the dots between millions of patent documents

#### Show your appreciation: If you find this project useful, please consider showing your support by starring it on GitHub. Your support means a lot! :star:

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Link-green.svg)](https://www.shamimahamed.com/patentmatch/)



# How to use the code
### 1. Download Data
Check `src/Data/README.md` file for instruction on how to setup dataset

### 2. Setup W&B api key
``` CLI
export WANDB_API_KEY=<your_key>
echo $WANDB_API_KEY
```

### 3. Training
``` CLI
python train.py --n_epochs 10 --batch_size 128 --learning_rate 0.0001
```
View train.py file for more detals


# License
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

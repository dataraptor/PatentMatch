# Write Assist - Predict

[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://www.shamimahamed.com/writeassist) &nbsp; [![View in W&B](https://img.shields.io/badge/View%20in-W%26B-blue)](https://wandb.ai/shamim/WriteAssist)




# Dataset


# Models
In this project, we used PyTorch and Hugging Face models such as BERT, RoBERTa, DeBERTa, etc. to develop proficiency models for the six analytic measures. We fine-tuned these models on the training set and made predictions on the test set to evaluate their performance.

# Results
Our best-performing model achieved an overall mean squared error (MSE) of 0.035 on the test set, with the following MSE scores for each of the six analytic measures:

- Cohesion: 0.042
- Syntax: 0.037
- Vocabulary: 0.030
- Phraseology: 0.034
- Grammar: 0.036
- Conventions: 0.035

## Weights and Biases Dashboard

You can view our training results and model performance in the Weights and Biases dashboard. 

Click the badge below to access the dashboard:

[![View in W&B](https://img.shields.io/badge/View%20in-W%26B-blue)](https://wandb.ai/<username>/<project_name>?workspace=user-<username>)







### Requirements
- Python 3.x
- PyTorch
- Huggingface Transformers
- Pandas

### Installation
1. Clone the repository:
``` bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

2. Install the required packages:
``` bash
pip install -r requirements.txt
```

### Data Preparation
1. Download the ELLIPSE corpus dataset from the competition website.
2. Extract the files and place them in the data/ folder of the repository.
3. Preprocess the `data/` by running the `preprocess.py` script
``` bash
python preprocess.py --data_dir data/ --output_dir data/processed
```
This will create preprocessed data files in the `data/processed` folder.

### Training
To train a model, run the `train.py` script with the desired model and hyperparameters. For example:
``` bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/processed/train.csv \
    --validation_file data/processed/valid.csv \
    --output_dir models/bert-base-uncased \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --metric_to_track "pearson" \
    --logging_steps 100 \
    --seed 42
```
This will train a BERT-based model on the preprocessed training and validation data, and save the best model to the `models/bert-base-uncased` folder.

### Evaluation
To evaluate a trained model, run the `evaluate.py` script with the path to the model and the test data. For example:
``` bash
python evaluate.py \
    --model_path models/bert-base-uncased \
    --test_file data/processed/test.csv \
    --output_dir output/bert-base-uncased
```
This will load the saved BERT-based model and evaluate it on the preprocessed test data, and save the predictions to the output/bert-base-uncased folder.

### Inference
To perform inference on new data using a trained model, run the predict.py script with the path to the model and the input data. For example:
``` bash
python predict.py \
    --model_path models/bert-base-uncased \
    --input_file input/essay.txt \
    --output_file output/predictions.csv
```
This will load the saved BERT-based model and predict the scores for the input essay, and save the predictions to the `output/predictions.csv` file.

### Note
- The hyperparameters mentioned in the training script are just an example, and you can experiment with different values for better performance.
- Make sure to change the model_name_or_path parameter in the training and evaluation scripts to the desired pre-trained model.


# Conclusion
Our results show that PyTorch and Hugging Face models can be effectively used to develop proficiency models for ELLs. These models can provide more accurate feedback on language development and expedite the grading cycle for teachers, which can ultimately lead to improved English language proficiency for ELLs.


# Citation
It is highly recommended to add citations in your research paper or project if you use this project for your experiments or results. Please note that proper citation is mandatory if you use this project for any research publication or project.

``` 
@misc{feedback-prize,
  author = {Your Name},
  title = {Improving Language Proficiency Prediction for English Language Learners with Deep Learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/feedback-prize}},
  commit = {INSERT COMMIT SHA HERE}
}
```

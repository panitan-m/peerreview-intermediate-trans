# Intermediate-Task Transfer Learning for Peer Review Score Prediction
Implementation of Intermediate-Task Transfer Learning for Peer Review Score Prediction.

## Environment Installation
1. Download and install [Anaconda](https://www.anaconda.com/products/individual)
2. Create an environment
```
conda env create -f environment.yml
conda activate peerreview-intermediate-trans
```

## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies and download some of the data files.
* [ASAP-Review](https://github.com/neulab/ReviewAdvisor) - Dataset for intermediate-task training (Review aspect sentiment prediction). <br />
  To download the dataset run,
  ```
  cd ReviewAdvisor
  sh download_dataset.sh
  ```
* [PeerRead](https://github.com/allenai/PeerRead/) - Dataset for target task fine-tuning (Review aspect score prediction).
* [PeerKit](https://github.com/panitan-m/peerkit) - Tool to manipulate PeerRead dataset.

## How-to-run
To train intermediate tasks and fine-tune on target tasks, please use the following command at the root of this repository:
```
./run.sh
```
The script first calls `train_intermediate.py` to train SciBERT on intermediate tasks (on ASAP-Review dataset) then calls `finetune.py` to fine-tune and evaluate the model (on PeerRead dataset).

Here is a brief description of each file.
* `intermediate_dataset.py` extracts review aspect sentiments from the ASAP-Review dataset and creates a dataset for intermediate-task training.
* `train_intermediate.py` trains SciBERT on review aspect sentiment prediction task as an intermediate task. You can specify the type of review aspect using the argument `--aspect`. The available types of review aspects are `clarity, meaningful_comparison, motivation, originality, soundness, substance`
* `finetune.py` fine-tunes a model after intermediate-task training and evaluates it using 5-fold cross-validation. You can specify the checkpoint of the model trained on intermediate task using the argument `--checkpoint`. You also can choose the type of review aspect for the score prediction using the argument `--aspects`. The available types of review aspects are `clarity, meaningful_comparison, impact, originality, recommendation, soundness_correctness, substance`.



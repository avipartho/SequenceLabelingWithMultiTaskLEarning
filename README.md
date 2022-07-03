# SequenceLabelingWithMultiTaskLearning
Multi-task Learning for sequence labeling tasks (e.g. NER).

Originally the code was developed for detecting social determinants of health (SDOH) from clinical notes. Given a text snippet, the task was to detect the SDOH type, presence and period of each token. Using this codebase, a BERT-based model can be trained for the three tasks simultaneously (multi-task learning) with a weighted loss function and we assume the dataset is in BIO tag format with each line representing the token, presence label, period label and SDOH label. However, it can be easily modified to use for other sequence labeling tasks.


## Requirements
This code was tested on `python3`. We suggest using a virtual environment.
* Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Run `conda env create -f environment.yml` to create an environment named `mtl_ner` with all necessary dependencies.
* Run `conda activate mtl_ner` to activate the conda environment.
* Create `datasets`, `logs`, and `output` directories inside the parent directory. Keep your data files inside the `datasets` directory.
  ```sh
  ❯ mkdir <dir_name>
  ```

## Train and Test
By default, the following shell script will train a `roberta-base` model and report results following an exact matching criterion. Once the training is complete using the default hyperparameters, the script will also report results for relaxed matching. For this, run the following command:
```sh
❯ bash run_ner.sh
```
Make sure to chage the data, model, output and log directories according to your work setup. The script supports multi-GPU training.

## Acknowledgements
* [BERT-NER repo](https://github.com/kamalkraj/BERT-NER)
* [Seqeval package](https://github.com/chakki-works/seqeval)
# Synth-SBDH
Data and Code to reproduce experiments with Synth-SBDH.

## Synth-SBDH Dataset
Synth-SBDH dataset, seed examples, prompt and data generation script will be released upon acceptance.

## Experiments
There are three directories for the three different experiments conducted in the paper - [mlc](mlc), [ner](ner) and [dss](dss). Each folder is self-contained with all the necessary scripts.

### Setting up development environment 

* Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Move to one of the experiment directories, for example, `cd mlc`
* Run `conda env create -f environment.yml` to create an environment
  with all necessary dependencies.
* Run `conda activate pyenv` to activate the conda environment.
* Create necessary directories inside the parent experiment directory. Check the bash scripts for this.
  ```sh
  mkdir <dir_name>
  ```
### Models
All models used in the three experiments are publicly available. Here is a list of those models - 

1. [RoBERTa-base](https://huggingface.co/roberta-base)
2. [ClinicalRoBERTa-base](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-hf.tar.gz)
3. [Mamba-130m](https://huggingface.co/state-spaces/mamba-130m)
4. [ClinicalMamba-130m]()
5. [T5-base](https://huggingface.co/google/t5-v1_1-base)
6. [FLAN-T5-base](https://huggingface.co/google/flan-t5-base)

### Datasets
We used the publicly available [MIMIC-SBDH](https://github.com/hibaahsan/MIMIC-SBDH) dataset to create MIMIC-SBDH~aligned~. This was used in the mlc experiment. For NER, we used a private dataset and unfortunately, we won't be able to share it.
Preprocessing notebooks for Synth-SBDH and MIMIC-SBDH are available in [data](data) directory.

### Running Experiments
--------
#### MLC
#### NER
#### DSS
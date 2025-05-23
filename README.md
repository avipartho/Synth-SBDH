# Synth-SBDH
Data and Codes to reproduce experiments in the paper **[Synth-SBDH: A Synthetic Dataset of Social and
Behavioral Determinants of Health for Clinical Text](https://arxiv.org/abs/2406.06056)**.
<!-- - [Synth-SBDH](#synth-sbdh)
  - [Synth-SBDH Dataset](#synth-sbdh-dataset)
  - [Experiments](#experiments)
    - [Setting up development environment](#setting-up-development-environment)
    - [Models](#models)
    - [Datasets](#datasets)
    - [Running Experiments](#running-experiments)
      - [MLC](#mlc)
      - [NER](#ner)
      - [DSS](#dss) -->

## Synth-SBDH Dataset
Synth-SBDH dataset, data generation scripts and fine-tuned model checkpoints will be released upon acceptance.

## Experiments
There are three directories for the three different experiments conducted in the paper - [mlc](mlc), [ner](ner) and [dss](dss). Each folder is self-contained with all the necessary scripts.

### Setting up development environment 

* Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Move to one of the experiment directories, for example, `cd mlc`
* Run `conda env create -f environment.yml` to create an environment
  with all necessary dependencies.
* Run `conda activate pyenv` to activate the conda environment.
* Create necessary directories (`logs`, `output`, `saved_models` etc.) inside the experiment directory. Run `mkdir <dir_name>`.
### Models
All models used in the three experiments are publicly available. Here is a list of those models - 

1. [RoBERTa-base](https://huggingface.co/roberta-base)
2. [ClinicalRoBERTa-base](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-hf.tar.gz)
3. [Mamba-130m](https://huggingface.co/state-spaces/mamba-130m)
4. [ClinicalMamba-130m](https://huggingface.co/whaleloops/clinicalmamba-130m-hf)
5. [T5-base](https://huggingface.co/google/t5-v1_1-base)
6. [FLAN-T5-base](https://huggingface.co/google/flan-t5-base)

### Datasets
We used the publicly available [MIMIC-SBDH](https://github.com/hibaahsan/MIMIC-SBDH) dataset to create MIMIC-SBDH<sub>aligned</sub>. This was used in the mlc experiment. For NER, we used a private dataset and unfortunately, we won't be able to share it.
Preprocessing notebooks for Synth-SBDH and MIMIC-SBDH wil be made available in the [data](data) directory.

### Running Experiments
--------
For all tasks, we do a two-stage supervised fine-tuning (SFT). In the first stage, we fine-tune models on the modified Synth-SBDH dataset (SFT<sub>stage1</sub>), and in the second stage, we use trained models from SFT<sub>stage1</sub> to further fine-tune on the task-specific real-world datasets (SFT<sub>stage2</sub>). Note that for DSS, there is no other real-world SBDH dataset with rationales, so there is no SFT<sub>stage2</sub>. 

Before running any of the following scripts, please update data path and all other neecessay parameters in the shell scripts.
#### MLC
1. Fine-tuning on Synth-SBDH (SFT<sub>stage1</sub>)
  ```sh
    cd mlc/e2e_scripts
    bash run_cliroberta.sh # ClinicalRoBERTa
  ```
2. Fine-tuning on MIMIC-SBDH<sub>aligned</sub> (SFT<sub>stage2</sub>)
  ```sh
    cd mlc/e2e_scripts
    bash run_cliroberta_2.sh # ClinicalRoBERTa
  ```
Similarly, for other models, run model-specific scripts.
#### NER
Fine-tuning on Synth-SBDH (SFT<sub>stage1</sub>)
  ```sh
    cd ner/e2e_scripts
    bash run_ner.sh # RoBERTa/ClinicalRoBERTa
  ```
For T5 and FLAN-T5 use `run_ner_t5.sh` script. The same scripts can be used for SFT<sub>stage2</sub> on any other dataset, given it follows similar data format.
#### DSS
Fine-tuning on Synth-SBDH (SFT<sub>stage1</sub>)
  ```sh
    cd dss/e2e_scripts/
    bash run_distilling_step_by_step_standard.sh # standard setup
    bash run_distilling_step_by_step_taskprefix.sh # with dss framework
  ```
## Citation
```
@misc{mitra2024synthsbdh,
      title={Synth-SBDH: A Synthetic Dataset of Social and Behavioral Determinants of Health for Clinical Text}, 
      author={Avijit Mitra and Emily Druhl and Raelene Goodwin and Hong Yu},
      year={2024},
      eprint={2406.06056},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
## Acknowledgments
Some of the scripts were taken and repurposed from the following repositories -
- https://github.com/whaleloops/ClinicalMamba
- https://github.com/google-research/distilling-step-by-step
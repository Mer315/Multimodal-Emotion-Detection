# Multimodal Emotion Detection on TESS

This project builds three emotion recognition pipelines on the Toronto Emotional Speech Set (TESS):

- speech-only
- text-only
- multimodal fusion of speech and text

The goal is to recognize the emotion label from:

- speech alone
- transcript text alone
- both modalities together

The seven target classes used throughout the code are:

- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `pleasant surprise`
- `sad`

## Dataset

The dataset used is the Toronto Emotional Speech Set (TESS), downloaded in the training scripts through `kagglehub`.

Each sample is mapped into:

- an audio path
- a speaker ID (`OAF` or `YAF`)
- a word
- an emotion label
- a transcript in the form `say the word <word>`

## Report and Experiment Artifacts

You can access the main experiment artifacts here:

- Google Drive folder for the project report, extracted features, and trained models:
  [Project Drive Folder](https://drive.google.com/drive/folders/1lucmFlJ9e5cVgKeMdtfYKR9-bmGnUwXr?usp=sharing)
- Local report PDF stored in this repository:
  [MULTIMODAL EMOTION RECOGNITION- report.pdf]()

## Project Structure

```text
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   ├── text_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   └── fusion_pipeline/
│       ├── train.py
│       └── test.py
├── Results/
│   ├── speaker_split_TSNE_plots.ipynb
│   └── word_level_split_TSNE_plots.ipynb
├── README.md
└── requirements.txt
```

Pretrained checkpoints currently stored in the repo:

- `models/BERT model/bert_fc_text_only_tess.pth`
- `models/Fusion model/hubert_bilstm_bert_latefusion_tess.pth`
- `models/HuBERT models/*.pth`
- `models/MFCC models/*.pth`

## Problem Formulation

The system is organized around the required functional blocks:

1. Preprocessing
2. Feature extraction
3. Temporal/contextual modelling
4. Fusion
5. Classification

## Architecture Decisions

### 1. Preprocessing

Speech preprocessing in the code includes:

- loading `.wav` files
- resampling to `16 kHz`
- handling variable-length sequences through padding/truncation
- fixed-size feature tensors for model input

Text preprocessing includes:

- constructing transcripts from the word labels
- lowercasing
- tokenization with BERT tokenizer
- padding/truncation to a fixed token length

### 2. Feature Extraction

Two speech feature families were explored:

- `MFCC + delta + delta-delta`
  - final frame-wise feature size: `120`
- `HuBERT base`
  - contextual speech embedding size: `768`

Text feature extraction uses:

- `BERT base uncased`
- CLS embedding of size `768`

### 3. Temporal / Contextual Modelling

Speech modelling experiments include:

- `CNN1D` over MFCC sequences
- `BiLSTM` over MFCC or HuBERT sequences
- `Multihead self-attention pooling`

Text modelling uses:

- a feed-forward classifier on top of BERT CLS embeddings

The final multimodal setup uses:

- `HuBERT + BiLSTM` for speech representation
- a linear projection for BERT text embeddings

### 4. Fusion

Fusion is implemented as late fusion:

- mean-pooled speech representation from BiLSTM
- projected text representation from BERT CLS
- concatenation of both vectors
- fully connected fusion layer
- final classifier head

### 5. Classifier

All pipelines end with a neural classifier head using combinations of:

- `Linear`
- `ReLU`
- `Dropout`

The final layer predicts one of the seven emotion classes.

## Pipelines

### Speech-only pipeline

Files:

- models/speech_pipeline/train.py
- models/speech_pipeline/test.py

What is implemented:

- TESS loading and label creation
- word-level split experiments
- speaker-level split experiments
- MFCC feature extraction
- HuBERT feature extraction
- model training for:
  - MFCC + CNN1D
  - MFCC + BiLSTM
  - MFCC + self-attention
  - HuBERT + CNN1D
  - HuBERT + BiLSTM
  - HuBERT + self-attention

Speaker-level speech experiments also include MFCC augmentation for robustness.

### Text-only pipeline

Files:

- models/text_pipeline/train.py
- models/text_pipeline/test.py

What is implemented:

- transcript construction from dataset metadata
- BERT tokenization
- BERT CLS feature extraction
- fully connected text classifier
- word-level split experiments
- speaker-level split experiments

### Fusion pipeline

Files:

- models/fusion_pipeline/train.py
- models/fusion_pipeline/test.py
  
What is implemented:

- loading precomputed HuBERT speech features
- loading precomputed BERT text features
- speech encoding with BiLSTM
- text projection with a linear layer
- late fusion through concatenation and a fusion fully connected block
- final multimodal classifier
- both word-level and speaker-level fusion evaluation code

## Data Split Strategy

Two split strategies appear in the repository.

### Word-level split

The word-level setup creates train, validation, and test partitions using disjoint words. Notes extracted from the project observations:

- cleaned dataframe size: `2800`
- speakers present: `OAF`, `YAF`
- unique words: `200`
- split sizes: `2240 / 280 / 280`
- overlap checks reported:
  - `Train ∩ Val = 0`
  - `Train ∩ Test = 0`
  - `Val ∩ Test = 0`

This split is useful for checking generalization across lexical content.

### Speaker-level split

The speaker-level setup separates speakers to reduce leakage:

- training speaker: `OAF`
- validation and test speaker: `YAF`, split 50/50
- split sizes used in code: `1400 / 700 / 700`

An important project observation was that an earlier random `80/10/10` split mixing both `YAF` and `OAF` caused leakage. The cleaner speaker-based split was introduced to evaluate real generalization better.

## Visualization and Analysis

The `Results/` folder contains scripts for t-SNE visualization of learned representations:

- Results/word_level_split_TSNE_plots.ipynb
- Results/speaker_split_TSNE_plots.ipynb

These scripts visualize cluster separability from:

- temporal modelling block
- contextual modelling block
- fusion block

The plotting code specifically covers:

- temporal representation t-SNE
  - `MFCC + CNN` in the speaker-level analysis
  - `HuBERT + BiLSTM` in the word-level analysis
- contextual representation t-SNE
  - `BERT CLS`
- fused representation t-SNE
  - `HuBERT + BiLSTM + BERT`

Based on the current scripts and observations, the analysis focus of this repository is:

- comparing unimodal vs multimodal performance
- checking whether fusion improves emotion separability
- understanding whether speaker leakage inflates performance
- visualizing how well emotion classes separate in latent space

## Results Summary

The repository includes trained checkpoints and evaluation scripts, but the final result tables are not yet stored as standalone files inside `Results/`.

Observed notes from the experiment log include:

- strong validation performance for several models on the word-level split
- cleaner evaluation after moving away from mixed-speaker random splitting
- dedicated speaker-level experiments for:
  - `MFCC + CNN`
  - `HuBERT + BiLSTM`
  - `BERT`
  - `HuBERT + BERT` fusion

To regenerate the final metrics, run the corresponding `test.py` files for each pipeline and save the printed classification reports into tables.

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

The training scripts download TESS from Kaggle using `kagglehub`.

Make sure your Kaggle access is configured correctly before running the scripts.

### 3. Run training

```bash
python models/speech_pipeline/train.py
python models/text_pipeline/train.py
python models/fusion_pipeline/train.py
```

### 4. Run evaluation

```bash
python models/speech_pipeline/test.py
python models/text_pipeline/test.py
python models/fusion_pipeline/test.py
```

### 5. Generate plots

```bash
python Results/word_level_split_TSNE_plots.ipynb
python Results/speaker_split_TSNE_plots.ipynb
```

## Important Implementation Note

The current codebase is written around Google Colab and Google Drive style paths such as:

- `/content/drive/MyDrive/...`

So before running locally, you will likely need to:

- update dataset and feature paths
- update checkpoint save/load paths
- remove or adapt `google.colab` drive mounting code

## Dependencies

Dependencies are listed in requirements.txt

Current third-party packages used in the repo include:

- `kagglehub`
- `librosa`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `soundfile`
- `torch`
- `tqdm`
- `transformers`


## Acknowledgment

This repository explores emotion recognition under both lexical and speaker generalization settings, with a strong emphasis on comparing:

- handcrafted vs pretrained speech features
- unimodal vs multimodal learning
- representation quality before and after fusion

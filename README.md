# DMSEdit
DMSEdit for semantic image editing


# Requirements

Install the required libraries.

```python
pip install -r requirements.txt
```

# Instalation

Download the best ViCHA checkpoint from https://github.com/mshukor/ViCHA and move it in the folder vicha.

Download the Stanford Parser from https://nlp.stanford.edu/software/lex-parser.shtml in the folder parser.

_Optional_: Generate new word alignments using https://github.com/chaojiang06/neural-Jacana.

# Data

Download the images, captions and alignments in the folder data.
- BISON 0.7
  - Images: [Link](https://drive.google.com/drive/folders/18RKSWSs42q3xq2Y6JHl4_AZKlnvxpJc4?usp=share_link)
  - Captions: [Link](https://drive.google.com/file/d/1mPOeQajLRzHRLS6DYiNUXLJNZjCTE6t4/view?usp=share_link)
  - Alignments: [Link](https://drive.google.com/file/d/1XVJGXNfjmAVapjPTSfr38O6OGJ_qpWOQ/view?usp=share_link)
- Dream
  - Images: [Link](https://drive.google.com/drive/folders/1RazlDU43B26N8HFZxBVYfqmerznZectH?usp=share_link)
  - Captions: [Link](https://drive.google.com/file/d/1fCEWqlJVgxw1ysPLEyMJ1yUCNdWUbOGo/view?usp=share_link)
  - Alignments: [Link](https://drive.google.com/file/d/1doxV4_65gE4RG8nrNZA9fUvTFEZCOJ2-/view?usp=share_link)

# Edit images with DMSEdit

BISON 07 dataset

```python
python /content/drive/MyDrive/dmsedit/run_dmsedit.py \
--path_input_data './data/bison_07.json'  \
--path_source_images './data/bison_07/' \
--path_alignments './data/alignments_bison_07.json'  \
--vicha_model_path './vicha/checkpoint_best.pth'  \
--bert_config_path './configs/config_bert.json'  \
--path_to_jar './parser/stanford-parser.jar' \
--path_to_models_jar './parser/stanford-parser-4.2.0-models.jar' \
--path_target_images "./output_bison_07/"  \
--token ' '  \
--cache_dir "./cache_dir/"
```

Dream dataset

```python
python /content/drive/MyDrive/dmsedit/run_dmsedit.py \
--path_input_data './data/dream.json'  \
--path_source_images './data/dream/' \
--path_alignments './data/alignments_dream.json'  \
--vicha_model_path './vicha/checkpoint_best.pth'  \
--bert_config_path './configs/config_bert.json'  \
--path_to_jar './parser/stanford-parser.jar' \
--path_to_models_jar './parser/stanford-parser-4.2.0-models.jar' \
--path_target_images "./output_dream/"  \
--token ' '  \
--cache_dir "./cache_dir/"
```
Note: token represents the Hugging Face token required for Stable Diffusion.

# Quick example

Add a new image in the folder examples and edit it using ```DMSEdit_example.ipynb ```

---
title: "CALCULUS - Symposium"
layout: text
excerpt: "CALCULUS Project at LIIR KU Leuven: Symposium"
sitemap: false
permalink: /symposium/
---

# The CALCULUS Symposium

#### Coming on January 29 - 30, 2024! 

More information follows soon. 

<div class="boxed">
### **Welcome**
The CALCULUS Symposium is the final workshop organized to celebrate the ending of the CALCULUS project coordinated by Prof. Marie-Francine Moens and funded by European Research Council (ERC) under the Horizon 2020 Advanced Grant. The workshop will be held in Leuven, Belgium between 29-30 January 2024.
CALCULUS Symposium aims to bring together researchers interested in developing machine learning models inspired by the human brain. It is an excellent opportunity for junior and senior researchers to learn about the work of the researchers involved in the CALCULUS project, to meet senior researchers involved in the field of machine learning and to present their recent work as a poster.
</div>





# DMSEdit
DMSEdit for semantic image editing


# Requirements

Install the required libraries/packages.

```python
pip install -r requirements.txt
```
```python
git clone https://github.com/facebookresearch/detectron2
```
```python
python -m spacy download en_core_web_md
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
--path_target_images './output_bison_07/'  \
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
--path_target_images './output_dream/'  \
--token ' '  \
--cache_dir "./cache_dir/"
```
Note: token represents the Hugging Face token required for Stable Diffusion.

# Quick example

Add a new image in the folder examples and edit it using ```DMSEdit_example.ipynb ```.

# Evaluation

Install the required libraries/packages.

```python
pip install lpips
```
```python
pip install pytorch_fid
```
```python
pip install git+https://github.com/openai/CLIP.git
```
Run the script ```evaluation_dmsedit.py```


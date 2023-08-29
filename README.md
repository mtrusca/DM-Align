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

#### Welcome 
The CALCULUS Symposium is the final workshop organized to celebrate the ending of the CALCULUS project coordinated by Prof. Marie-Francine Moens and funded by European Research Council (ERC) under the Horizon 2020 Advanced Grant. The workshop will be held in Leuven, Belgium between 29-30 January 2024.

CALCULUS Symposium aims to bring together researchers interested in developing machine learning models inspired by the human brain. It is an excellent opportunity for junior and senior researchers to learn about the work of the researchers involved in the CALCULUS project, to meet senior researchers involved in the field of machine learning and to present their recent work as a poster.

#### Venue
The event will take place at the Arenberg Castle (Arenbergkasteel) in Leuven. 

Location: Kardinaal Mercierlaan 94, 3001 Leuven

#### Registration
Please use this form [we will add the form for registration] to register.

Fees: …..

#### Schedule
##### Monday, 29th: Human-inspired learning
12:00-12:30: Registration
12:30-13:00: Reception
13:00-14:00: Keynote speaker 1
14:00-14:30: Coffee break + posters
14:00-15:30: Internal Speakers - First Session
16:00-16:30: Coffee break + posters
16:30-18:00: Internal speakers - Second Session

##### Tuesday, 30th: Multi-modal learning
8:30-9:45: Registration
8:45-9:00: Reception 
9:00-10:00: Keynote speaker 2
10:00-10:15: Coffee break
10:15-12:00: Internal speakers - First Session
12:00-13:00: Lunch break + posters
13:00-14:00: Keynote speaker 3
14:00-14:15: Coffee break
14:15-15:45: Internal speakers - Second Session
15:45 - 16:00 Ending … talk (organizers?)

#### Submission Guidelines
The researchers are invited to submit a single-page abstract of either published work or ongoing work following this template [we will add the template for submissions]. 

During the symposium, the abstracts will be presented as posters. The abstracts should be related to the goal of the CALCULUS project based on the following topics:
[list of topics].

### Each submission should contain: 
- Title of the presentation.
- Name and the affiliation of the authors.
- Abstract
- References
  
The abstracts should be submitted to: calculus-sym@cs.kuleuven.be. There are no printed proceedings.

### Important Dates:

Submission Deadline: 
Notification of acceptance:
Early registration deadline: 





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


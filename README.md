# DM-Align
DM-Align for semantic image editing

# Requirements

Install the required libraries/packages.

```
pip install -r requirements.txt
```
```
python -m spacy download en_core_web_md
```

# Instalation

Download the Stanford Parser from https://nlp.stanford.edu/software/lex-parser.shtml in the folder parser.

Download spanBERT and the pre-trained checkpoint for word alignments from https://github.com/chaojiang06/neural-Jacana in the folder models.

Generate new word alignments between source and target text instructions using:

```
python generate_word_alignments.py \
--path_data 'data/json_files/dream.json' \
--path_alignments 'data/json_files/alignments_dream.json'
```
Note: running the file generate_word_alignments.py requires transformers 2.2.2.

Extract objects associated with the keywords of the source and target text instructions using:

```
python grounded_sam_objects.py \
--path_data 'data/json_files/dream.json' \
--path_alignment 'data/json_files/alignments_dream.json' \
--path_images 'data/dream' \
--path_objects 'data/dream_objects'
```

# Data

Download the input images in the folder data.
- Imagen
  - Images: [Link](https://drive.google.com/drive/folders/1mKHLljrOGHAkGAHdeW89b_mG5C7H34x3?usp=sharing)
- Dream
  - Images: [Link](https://drive.google.com/drive/folders/1RazlDU43B26N8HFZxBVYfqmerznZectH?usp=sharing)

# Edit images with DM-Align

```
python dm_align.py \
--path_data 'data/json_files/dream.json'  \
--path_images 'data/dream' \
--path_alignment 'data/json_files/alignments_dream.json'  \
--path_objects 'data/dream_objects'  \
--token_huggingface ''  \
--output_dir 'results/dream' \
--save_masks 1 \
--path_mask 'data/dream_masks'
```
Note: token_huggingface represents the Hugging Face token required for Stable Diffusion.

# Evaluation

Run the script 
```
python evaluation.py \
--path_data 'data/json_files/dream.json'  \
--output_dir 'results/dream' \
--path_mask 'data/dream_masks' \
--path_images 'data/dream' \
--back 0
```
 Note: use back 1 to evaluate only the background.

import io
import os
import sys
import torch
import warnings
import numpy as np
from PIL import Image
from copy import deepcopy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
sys.path.insert(0, os.path.abspath('/content/detectron2'))
from detectron2.data import MetadataCatalog, DatasetCatalog
from transformers import DetrFeatureExtractor, DetrForSegmentation
warnings.filterwarnings("ignore")

def load_detr():
  feature_extractor_detr = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic",
                                                                cache_dir  = 'new_cache_dir')
  model_detr = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic",
                                                   cache_dir  = 'new_cache_dir')
  return feature_extractor_detr, model_detr

def rgb_to_id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def detr_mask(result_detr, words, indexes):
  meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
  segments_info = deepcopy(result_detr["segments_info"])
  panoptic_seg = Image.open(io.BytesIO(result_detr['png_string'])).resize((512, 512))
  final_w, final_h = panoptic_seg.size
  panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
  panoptic_seg = torch.from_numpy(rgb_to_id(panoptic_seg))

  temp_dict = {}
  for i, w in enumerate(words):
    temp_dict[w] = indexes[i]
  wordnet_lemmatizer = WordNetLemmatizer()

  detected_obj, imgs, detected_obj_idx= [], [], []
  for i in range(len(segments_info)):
      c = segments_info[i]["category_id"]
      segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]
      obj = meta.thing_classes[segments_info[i]["category_id"]] if segments_info[i]["isthing"] else meta.stuff_classes[segments_info[i]["category_id"]]
      try:
        obj_class = wordnet.synsets(obj)[0]
        hypo_obj = set([i for i in obj_class.closure(lambda s:s.hyponyms())])
      except:
        obj = ' '.split(obj)[-1]
        print("An exception occurred", obj)
        hypo_obj = []
      for word in words:
          word_lemma = wordnet_lemmatizer.lemmatize(word)
          synonym = []
          for syn in wordnet.synsets(word_lemma):
              for l in syn.lemmas():
                  lemma_word = wordnet.synsets(word)[0]
                  if len(hypo_obj) >0:
                      if l.name() == obj or lemma_word in hypo_obj:
                          imgs.append(np.array((panoptic_seg == segments_info[i]["id"]).float() * 1))
                          detected_obj.append(word)
                          detected_obj_idx.append(temp_dict[word])
                  else:
                      if l.name() == obj:
                          imgs.append(np.array((panoptic_seg == segments_info[i]["id"]).float() * 1))
                          detected_obj.append(word)
                          detected_obj_idx.append(temp_dict[word])

  final_mask = np.maximum(0, np.minimum(1, sum(imgs)))
  final_mask = final_mask.astype(np.uint8)
  return final_mask, [i for i in words if i not in detected_obj], [i for i in indexes if i not in detected_obj_idx]


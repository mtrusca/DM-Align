
import numpy as np
import torch, logging
from PIL import Image
logging.disable(logging.WARNING)
import os, json, gc, ast, pathlib

from detr import detr_mask
from vicha_gradcam import vicha_gradcam_mask
from keywords import detect_noun_relation, extract_words_alignment
from diffusion import create_mask_diffusion, improve_mask, reshape_mask, dmsedit_diffusion

def dmsedit(path_image, source_caption, target_caption, alignment,
                scheduler, text_encoder, tokenizer, unet, vae, pipe_diffusion,
                vicha_tokenizer, vicha_model,
                feature_extractor_detr, model_detr,
                dependency_parser, wordnet_lemmatizer):
  dim1, dim2 = np.array(Image.open(path_image).convert('RGB')).shape[:2]
  init_img_512 = Image.open(path_image).convert('RGB').resize((512,512))
  # alignment = ast.literal_eval(alignments[obj['image_filename']])
  noun_modifiers_pairs1, noun_modifiers_pairs2, noun_verb_pairs = detect_noun_relation(source_caption,
                                                                                        target_caption,
                                                                                        dependency_parser)
  add1, add2, remove1, add1_tok, add2_tok, remove1_tok = extract_words_alignment(source_caption,
                                                                                  target_caption, alignment,
                                                                                  noun_modifiers_pairs1,
                                                                                  noun_modifiers_pairs2,
                                                                                  noun_verb_pairs,
                                                                                  wordnet_lemmatizer)
  torch.cuda.empty_cache()
  gc.collect()

  mask = create_mask_diffusion(scheduler, text_encoder, tokenizer, unet, vae, init_img=init_img_512,
                          rp=source_caption, qp=target_caption, n=20)
  mask = (mask > 0).astype("uint8")
  mask = improve_mask(mask)
  torch.cuda.empty_cache()
  gc.collect()

  # detr
  encoding_detr = feature_extractor_detr(init_img_512, return_tensors="pt")
  outputs_detr = model_detr(**encoding_detr)
  processed_sizes_detr = torch.as_tensor(encoding_detr['pixel_values'].shape[-2:]).unsqueeze(0)
  result_detr = feature_extractor_detr.post_process_panoptic(outputs_detr, processed_sizes_detr)[0]
  detr_mask_add, detr_add1_new, detr_add1_new_idx = detr_mask(result_detr, add1_tok, add1)
  detr_mask_remove, detr_remove1_new, detr_remove1_new_idx = detr_mask(result_detr, remove1_tok,
                                                                              remove1)
  del encoding_detr, outputs_detr, processed_sizes_detr, result_detr

  if len(detr_mask_add.shape) > 0:
      detr_mask_add = reshape_mask(mask, detr_mask_add)
  if len(detr_mask_remove.shape) > 0:
      detr_mask_remove = reshape_mask(mask, detr_mask_remove)

  # vicha
  torch.cuda.empty_cache()
  gc.collect()
  detr_mask_remove_vicha_step2, detr_mask_add_vicha_step2, _, _, _, _ = vicha_gradcam_mask(path_image, source_caption,
                                                                                  target_caption,
                                                                                  detr_add1_new_idx, add2,
                                                                                  detr_remove1_new_idx,
                                                                                  vicha_model, vicha_tokenizer,
                                                                                  0.10)

  if len(detr_mask_add_vicha_step2.shape) > 0:
      detr_mask_add_vicha_step2 = reshape_mask(mask, detr_mask_add_vicha_step2)
  if len(detr_mask_remove_vicha_step2.shape) > 0:
      detr_mask_remove_vicha_step2 = reshape_mask(mask, detr_mask_remove_vicha_step2)

  if len(detr_mask_remove.shape) > 0:
      mask_remove_detr_vicha = np.maximum(0, np.minimum(1, detr_mask_remove + detr_mask_remove_vicha_step2))
  else:
      mask_remove_detr_vicha = detr_mask_remove_vicha_step2
  if len(detr_mask_add.shape) > 0:
      mask_add_detr_vicha = np.maximum(0, np.minimum(1, detr_mask_add + detr_mask_add_vicha_step2))
  else:
      mask_add_detr_vicha = detr_mask_add_vicha_step2
  torch.cuda.empty_cache()
  gc.collect()
  del detr_mask_remove, detr_mask_remove_vicha_step2, detr_mask_add, detr_mask_add_vicha_step2

  mask5, output_mask_dmsedit = dmsedit_diffusion(scheduler, text_encoder, tokenizer, unet, vae, pipe_diffusion, init_img_512,
                              rp = [source_caption], qp=[target_caption], mask=mask, mask_in=mask_remove_detr_vicha,
                              mask_out=mask_add_detr_vicha)
  torch.cuda.empty_cache()
  gc.collect()

  output_mask_dmsedit[0] = output_mask_dmsedit[0].resize((dim2, dim1))

  return output_mask_dmsedit[0]
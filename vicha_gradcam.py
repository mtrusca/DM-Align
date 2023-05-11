import json
import torch, re
import numpy as np
from torch import nn
from PIL import Image
from functools import partial
from scipy.ndimage import filters
from torchvision import transforms
import matplotlib.pyplot as plt
from models.vit import VisionTransformer
from models.vit import interpolate_pos_embed
from models.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer
from skimage import transform as skimage_transform

def vicha_gradcam_mask(image_path, caption, caption1, add1, add2, remove1, model, tokenizer, tr):
    image_pil = Image.open(image_path).convert('RGB').resize((512, 512))
    gradcam_in, text_vis_in = generate_gradcam(image_pil, caption, caption1, model, 'itm', tokenizer)
    gradcam_out, text_vis_out = generate_gradcam(image_pil, caption1, caption, model, 'itm', tokenizer)
    gradcam_new_in = clean_gradcam(gradcam_in, text_vis_in, tokenizer)
    gradcam_new_out = clean_gradcam(gradcam_out, text_vis_out, tokenizer)
    mask_in = map_word_gradcam(remove1, gradcam_new_in, tr)
    mask_out1 = map_word_gradcam(add1, gradcam_new_in, tr)
    mask_out2 = map_word_gradcam(add2, gradcam_new_out, tr)
    mask_out = np.maximum(0, np.minimum(1, mask_out1 + mask_out2))
    return mask_in, mask_out, text_vis_in, text_vis_out, gradcam_in, gradcam_out

def prepare_word_mask(attMap, tr):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, ((512, 512)), order=3, mode='constant')
    attMap = filters.gaussian_filter(attMap, 0.02 * max((512, 512)))
    attMap -= attMap.min()
    attMap /= attMap.max()
    attMap = np.array(attMap > tr)
    return attMap

def map_word_gradcam(indices, gradcam, tr):
    mask = torch.zeros(24, 24)
    mask = skimage_transform.resize(mask, ((512, 512)), order=3, mode='constant')
    mask = np.array(mask > tr)
    for ind in indices:
        attMap = prepare_word_mask(gradcam[ind + 1], tr)
        mask = np.logical_or(mask, attMap)
    attMap = np.array(mask * 1, dtype=np.uint8)
    return attMap

def clean_gradcam(gradcam, text_vis, tokenizer):
    for i, token_id in enumerate(text_vis.input_ids[0][1:]):
        token = tokenizer.decode([token_id])
        if token[:2] == '##':
            if i == len(text_vis.input_ids[0][1:]):
                gradcam = gradcam[:i]
            else:
                gradcam = torch.cat((gradcam[:i], gradcam[i + 1:]))
    return gradcam

def pre_caption(caption, max_words=30):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption, )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
            attMap.shape + (1,)) * attMapV
    return attMap

def generate_gradcam(image_pil, caption, caption1, model, gradcam_mode, tokenizer):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    image = transform(image_pil).unsqueeze(0)
    text = pre_caption(caption)
    text1 = pre_caption(caption1)
    image = image.cuda()
    text = tokenizer(text, return_tensors="pt")
    text = text.to(image.device)
    text1 = tokenizer(text1, return_tensors="pt")
    text1 = text1.to(image.device)
    if gradcam_mode == 'itm':
        output = model(image, text, text1)
        loss = output[:, 1].sum()
    elif gradcam_mode == 'itc':
        image_feat, text_feat = model(image, text, text1, itc=True)
        sim = image_feat @ text_feat.t() / model.temp
        loss = sim.diag().sum()
    model.zero_grad()
    loss.backward()
    text_vis = text
    patch_index = 577
    block_num = 8
    feat_map_res = 24
    if gradcam_mode == 'itc':
        with torch.no_grad():
            grad = model.visual_encoder.blocks[block_num].attn.get_attn_gradients().detach()
            cam = model.visual_encoder.blocks[block_num].attn.get_attention_map().detach()
            cam = cam[:, :, 0, 1:patch_index].reshape(image.size(0), -1, feat_map_res, feat_map_res)
            grad = grad[:, :, 0, 1:patch_index].reshape(image.size(0), -1, feat_map_res, feat_map_res).clamp(0)
            gradcam = (cam * grad).mean(1)
    else:
        with torch.no_grad():
            mask = text_vis.attention_mask.view(text_vis.attention_mask.size(0), 1, -1, 1, 1)
            grads = model.text_encoder.base_model.base_model.encoder.layer[
                block_num].crossattention.self.get_attn_gradients()
            cams = model.text_encoder.base_model.base_model.encoder.layer[
                block_num].crossattention.self.get_attention_map()
            cams = cams[:, :, :, 1:patch_index].reshape(image.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:patch_index].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask
            gradcam = cams * grads
            gradcam = gradcam[0].mean(0).cpu().detach()
    return gradcam, text_vis

class kw_img_ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert='',
                 img_size=380, ):
        super().__init__()
        bert_config = BertConfig.from_json_file(config_bert)
        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), return_attention=True)
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
        bert_config_kw = BertConfig.from_json_file(config_bert)
        self.num_hidden_layers_kw = 2
        bert_config_kw.num_hidden_layers = self.num_hidden_layers_kw
        bert_config_kw.fusion_layer = self.num_hidden_layers_kw
        text_width = self.text_encoder.config.hidden_size
        vision_width = 768
        self.kw_encoder = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False)
        self.kw_proj = nn.Linear(text_width, vision_width)
        embed_dim = 256
        self.itm_head = nn.Linear(768, 2)
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]))

    def forward(self, image, text, kwords, itc=False, block_num=-1):
        # text, kwords = text
        ## kw
        kw_output = self.kw_encoder(kwords.input_ids, attention_mask=kwords.attention_mask,
                                    return_dict=True, mode='text')
        kw_embeds = kw_output.last_hidden_state
        kw_embeds = self.kw_proj(kw_embeds)
        kw_embeds_external = kw_embeds
        image_embeds = self.visual_encoder(image, external_features=kw_embeds_external, register_blk=block_num)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        if itc:
            image_feat = self.vision_proj(image_embeds[:, 0, :])
            image_feat = F.normalize(image_feat, dim=-1)

            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
            return image_feat, text_feat
        else:
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True, )
            vl_embeddings = output.last_hidden_state[:, 0, :]
            vl_output = self.itm_head(vl_embeddings)
            return vl_output

def load_vicha(model_path, bert_config_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # , local_files_only=True)
    model = kw_img_ALBEF(text_encoder='bert-base-uncased', config_bert= bert_config_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint["model"]
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    msg = model.load_state_dict(state_dict, strict=False)
    model.eval()
    block_num = 8
    model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
    model.cuda()
    return tokenizer, model

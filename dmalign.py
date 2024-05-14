import gc, string, pathlib, ast, nltk, spacy, json, argparse, os, torch, logging, cv2
import numpy as np
from PIL import Image
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler , StableDiffusionInpaintPipeline
logging.disable(logging.WARNING)
from torchvision import transforms as tfms
from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet')

def reshape_mask(mask, mask1):
    dim10, dim20 = mask.shape
    dim11, dim21 = mask1.shape
    m1 = mask1[::int(dim11/dim10), ::int(dim21/dim20)]
    m1 = m1[:dim10, :dim20]
    return m1

def reshape_mask_with_dim(mask1, dim10, dim20):
    dim11, dim21 = mask1.shape
    m1 = mask1[::int(dim11/dim10), ::int(dim21/dim20)]
    m1 = m1[:dim10, :dim20]
    return m1

def detect_noun_relation(source, target, dependency_parser):
    doc1 = nlp(source.lower())
    doc2 = nlp(target.lower())
    tokens1 = nltk.word_tokenize(source.lower())
    tokens2 = nltk.word_tokenize(target.lower())
    result = dependency_parser.raw_parse(target)
    dep =next(result)
    objects1, objects2 = [], []
    noun_modifiers_pairs1 = {}
    for chunk in doc1.noun_chunks:
        tok_l = nlp(chunk.text).to_json()['tokens']
        for t in tok_l:
            head = tok_l[t['head']]
        objects1.append(chunk.text[head['start']:head['end']])
        modifiers_idx = []
        noun = ""
        for tok in chunk:
            if tok.pos_ == "NOUN" and str(tok) in objects1:
                noun = tok.text
                noun_idx = tokens1.index(str(noun))
            if tok.pos_ == "ADJ" or tok.pos_ == 'NUM' or (tok.pos_ =='NOUN' and str(tok) not in objects1):
                modifiers_idx.append(tokens1.index(str(tok.text)))
        if noun:
            noun_modifiers_pairs1.update({noun_idx:modifiers_idx})
    noun_modifiers_pairs2 = {}
    for chunk in doc2.noun_chunks:
        tok_l = nlp(chunk.text).to_json()['tokens']
        for t in tok_l:
            head = tok_l[t['head']]
        objects2.append(chunk.text[head['start']:head['end']])
        modifiers_idx = []
        noun = ""
        for tok in chunk:
            if tok.pos_ == "NOUN" and str(tok) in objects2:
                noun = tok.text
                noun_idx = tokens2.index(str(noun))
            if tok.pos_ == "ADJ" or tok.pos_ == 'NUM' or (tok.pos_ =='NOUN' and str(tok) not in objects2):
                modifiers_idx.append(tokens2.index(str(tok.text)))
        if noun:
            noun_modifiers_pairs2.update({noun_idx:modifiers_idx})
    noun_verb_pairs ={}
    verbs = []
    for trip in list(dep.triples()):
        if trip[0][0] in objects2 and trip[0][1] in ['NN', 'NNS'] and trip[2][1] == 'VBG' and trip[1] =='acl':
            noun_verb_pairs[tokens2.index(trip[0][0])] = tokens2.index(trip[2][0])
            verbs.append(trip[2][0])
        if trip[2][0] in objects2 and trip[2][1] in ['NN', 'NNS'] and trip[0][1] == 'VBG' and trip[1] == 'nsubj':
            noun_verb_pairs[tokens2.index(trip[2][0])] = tokens2.index(trip[0][0])
            verbs.append(trip[0][0])
    for trip in list(dep.triples()):
        if trip[0][0] in verbs and trip[0][1] == 'VBG' and trip[2][1] == 'VBG' and (trip[1] == 'conj' or trip[1] == 'advcl') and \
            len([x for x in dep.triples() if x[0] == trip[2] and x[2][1] in ['NN', 'NNS'] and x[1] == 'nsubj']) == 0:
            noun_key = [i for i in noun_verb_pairs if noun_verb_pairs[i] == tokens2.index(trip[0][0]) ][0]
            temp_values = noun_verb_pairs[noun_key] if isinstance(noun_verb_pairs[noun_key], list) else [noun_verb_pairs[noun_key]]
            temp_values.append(tokens2.index(trip[2][0]))
            noun_verb_pairs[noun_key] = temp_values

    return noun_modifiers_pairs1, noun_modifiers_pairs2, noun_verb_pairs

def extract_words_alignment(source, target, alignment, noun_modifiers_pairs1, noun_modifiers_pairs2, noun_verb_pairs, wordnet_lemmatizer):
    doc1 = nlp(source.lower())
    doc2 = nlp(target.lower())
    tokens1 = nltk.word_tokenize(source.lower())
    tokens2 = nltk.word_tokenize(target.lower())
    add1, remove1, add2 = [], [], []
    alignment_sent1 = list(set([int(i.split('-')[0]) for i in alignment[0]]))
    alignment_sent2 = list(set([int(i.split('-')[1]) for i in alignment[0]]))
    for pair in list(alignment[0]):
        ind1, ind2 = [int(i) for i in pair.split('-')]
        lemma1 = wordnet_lemmatizer.lemmatize(tokens1[ind1])
        lemma2 = wordnet_lemmatizer.lemmatize(tokens2[ind2])
        lemma1_syn = list(set([word.name() for syn in wordnet.synsets(lemma1) for word in syn.lemmas()]))
        lemma2_syn = list(set([word.name() for syn in wordnet.synsets(lemma2) for word in syn.lemmas()]))
        # updated part
        if lemma1 not in lemma2_syn and lemma2 not in lemma1_syn and lemma1 != lemma2:
            if doc1[ind1].pos_ == 'NOUN' and doc2[ind2].pos_ == 'NOUN':
                add1.append(ind1)
                add2.append(ind2)
        # property changed
        if lemma1 in lemma2_syn or lemma2 in lemma1_syn or lemma1 == lemma2:
            if doc1[ind1].pos_ == 'NOUN' and doc2[ind2].pos_ == 'NOUN':
                temp1, temp2 = [], []
                if ind1 in noun_modifiers_pairs1.keys():
                    temp1 = [tokens1[i] for i in noun_modifiers_pairs1[ind1]]
                if ind2 in noun_modifiers_pairs2.keys():
                    temp2 = [tokens2[i] for i in noun_modifiers_pairs2[ind2]]
                if sorted(temp1) == sorted(temp2) or len(temp2) == 0:
                    remove1.append(ind1)
                else:
                    add1.append(ind1)
            # inserted part
            if doc2[ind2].pos_ == 'NOUN':
                if ind2 in noun_verb_pairs.keys():
                    if isinstance(noun_verb_pairs[ind2], list):
                        for vb in noun_verb_pairs[ind2]:
                            if vb not in alignment_sent2: add2.append(vb)
                    else:
                        if noun_verb_pairs[ind2] not in alignment_sent2: add2.append(noun_verb_pairs[ind2])

    # deleted part
    for i in range(len(tokens1)):
        if i not in alignment_sent1:
            if doc1[i].pos_ == 'NOUN':
              # remove1.append(i)  # pastreaza obiectul
              add1.append(i)  #elimina obiectul
    add1 = list(set(add1))
    add2 = list(set(add2))
    remove1 = list(set(remove1))
    add1_tok = [tokens1[i] for i in add1]
    add2_tok = [tokens2[i] for i in add2]
    remove1_tok = [tokens1[i] for i in remove1]
    return add1, add2, remove1, add1_tok, add2_tok, remove1_tok

def load_models(token, cache_dir):
    YOUR_TOKEN =token
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16, cache_dir=cache_dir).to("cuda")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                              cache_dir=cache_dir,
                                              use_auth_token=YOUR_TOKEN,
                                              subfolder="unet", torch_dtype=torch.float16).to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",cache_dir=cache_dir,revision="fp16",torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN).to("cuda")
    return scheduler, text_encoder, tokenizer, unet, vae, pipe

def dm_align(pipe, p, qp, mask, mask_remove, mask_add):
    if len(mask_remove.shape) > 0:
      mask_in = reshape_mask(mask, mask_remove)
      mask = np.minimum(1, np.maximum(0, mask.astype('int64') - mask_in.astype('int64')))
    if len(mask_add.shape) > 0:
      mask_out = reshape_mask(mask, mask_add)
      mask = np.maximum(0, np.minimum(1, mask.astype('int64') + mask_out.astype('int64')))
    mask = mask.astype('uint8')
    output = pipe(prompt=qp,
        image=Image.open(p).convert('RGB').resize((512,512)),
        mask_image=Image.fromarray(mask * 255).resize((512, 512)),
        generator=torch.Generator("cuda").manual_seed(100),
        num_inference_steps=50).images
    input_size = np.array(Image.open(p).convert('RGB')).shape[:2]
    output[0] = output[0].resize([input_size[1], input_size[0]])
    return mask, output

def pil_to_latents(vae, image):
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist

def encode_text(text_encoder, tokenizer, prompts, maxlen=None):
    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to("cuda"))[0].half()

def caption_to_img_noise(scheduler, text_encoder, tokenizer, unet, vae, prompts, init_img, final_mask_remove, caption_type = 'target',
                          map_error = True, g=7.5, seed=100, strength=0.3, steps=50, dim=512):
    text = encode_text(text_encoder, tokenizer, prompts)
    uncond = encode_text(text_encoder, tokenizer, [""], text.shape[1])
    if caption_type == 'target':
        uncond = encode_text(text_encoder, tokenizer, [""], text.shape[1])
    if caption_type == 'source':
        uncond = encode_text(text_encoder, tokenizer, prompts)
    emb = torch.cat([uncond, text])
    if seed: torch.manual_seed(seed)
    scheduler.set_timesteps(steps)
    init_latents = pil_to_latents(vae, init_img)
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)

    if map_error:
        m1 = reshape_mask_with_dim(final_mask_remove, noise.shape[2], noise.shape[3])
        all_noises = []
        for i in range(noise.shape[1]):
          temp_noise = noise[0][i]
          temp_noise[np.where(m1 ==1)] = 0
          all_noises.append(temp_noise)
        masked_noise = torch.stack(all_noises, axis =0)
        masked_noise = torch.unsqueeze(masked_noise, axis=0)
        masked_noise = masked_noise.type(init_latents.dtype)
        masked_noise = masked_noise.cuda()
        noise = masked_noise

    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timesteps)
    with torch.no_grad(): u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)
    pred = u + g * (t - u)
    latents = scheduler.step(pred, timesteps, latents).pred_original_sample
    return latents.detach().cpu()

def create_mask(scheduler, text_encoder, tokenizer, unet, vae, init_img, rp, qp, final_mask_remove, map_error, n=20, s=0.5):
    diff, d1, d2 = {}, {}, {}
    for idx in range(n):
        orig_noise = caption_to_img_noise(scheduler, text_encoder, tokenizer, unet, vae,
                                           prompts=rp, init_img=init_img, final_mask_remove = final_mask_remove, map_error=map_error,
                                           caption_type = 'source',
                                           strength=s, seed=100 * idx)[0]
        query_noise = caption_to_img_noise(scheduler, text_encoder, tokenizer, unet, vae,
                                            prompts=qp, init_img=init_img, final_mask_remove = final_mask_remove, map_error=map_error,
                                            caption_type = 'target',
                                            strength=s, seed=100 * idx)[0]
        diff[idx] = (np.array(orig_noise) - np.array(query_noise))
    mask = np.zeros_like(diff[0])
    for idx in range(n):
        mask += np.abs(diff[idx])
    mask = mask.mean(0)
    mask = (mask - mask.mean()) / np.std(mask)
    return mask

def improve_mask(mask):
    mask  = cv2.GaussianBlur(mask*255,(3,3),1) > 0
    return mask.astype('uint8')

def run(args):
    path_to_jar = args.path_to_jar
    path_to_models_jar = args.path_to_models_jar
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    wordnet_lemmatizer = WordNetLemmatizer()
    scheduler, text_encoder, tokenizer, unet, vae, pipe = load_models(args.token_huggingface, args.cache_dir)

    path_data = args.path_data
    with open(path_data, 'r') as myfile: data = myfile.read()
    data = json.loads(data)
    path_alignment = args.path_alignment
    with open(path_alignment, 'r') as myfile: alignments = myfile.read()
    alignments = json.loads(alignments)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for i, obj in tqdm(enumerate(data)):
        p = os.path.join(args.path_images, obj['image_filename'])
        obj['caption1'] = obj['caption1'].strip().lower()#.replace('.','').replace("'s", 's')
        obj['caption2']=obj['caption2'].strip().lower()#.replace('.','').replace("'s", 's')
        obj['caption1'] = obj['caption1'].translate(str.maketrans('', '', string.punctuation))
        obj['caption2'] = obj['caption2'].translate(str.maketrans('', '', string.punctuation))
        k = pathlib.Path(obj['image_filename']).stem
        init_img_512 = Image.open(p).convert('RGB').resize((512,512))
        if os.path.isfile(p):
            alignment = ast.literal_eval(alignments[obj['image_filename']])
            noun_modifiers_pairs1, noun_modifiers_pairs2, noun_verb_pairs= detect_noun_relation(obj['caption1'], obj['caption2'],
                                                                                                dependency_parser)
            add1, add2, remove1, add1_tok, add2_tok, remove1_tok = extract_words_alignment(obj['caption1'], obj['caption2'], alignment,
                                                                  noun_modifiers_pairs1, noun_modifiers_pairs2,
                                                                  noun_verb_pairs, wordnet_lemmatizer)
            final_mask_img, mask, init_mask_add, init_mask_remove = np.zeros([64, 64]), np.zeros([64, 64]), np.zeros([64, 64]), np.zeros([64, 64])
            if args.with_objects_sam == 1:
                if remove1_tok != []:
                    try:
                        path_objects = os.path.join(args.path_objects, k + '_remove.npy')
                        with open(path_objects, 'rb') as f: maps = np.load(f)
                        init_mask_remove = reshape_mask_with_dim(maps, 64, 64)
                        init_mask_remove[init_mask_remove > 0] = 1
                    except:
                        init_mask_remove = np.zeros([64, 64])
                if add1_tok != []:
                    try:
                        path_objects = os.path.join(args.path_objects, k + '_add.npy')
                        with open(path_objects, 'rb') as f: maps = np.load(f)
                        init_mask_add = reshape_mask_with_dim(maps, 64, 64)
                        init_mask_add[init_mask_add > 0] = 1
                    except:
                        init_mask_add = np.zeros([64, 64])
            else:
                init_mask_add, init_mask_remove = np.zeros([64, 64]), np.zeros([64, 64])

            torch.cuda.empty_cache()
            gc.collect()
            if args.with_diffusion == 1:
                mask = create_mask(scheduler, text_encoder, tokenizer, unet, vae,
                                        init_img=init_img_512, rp=obj['caption1'], qp=obj['caption2'],
                                        final_mask_remove = init_mask_remove, map_error = args.with_noise_cancellation,
                                        n=20)
            else:
                mask = np.zeros([64, 64])
            mask = (mask > 0).astype("uint8")
            mask = improve_mask(mask)
            torch.cuda.empty_cache()
            gc.collect()
            final_mask_img, output = dm_align(pipe, p, qp=[obj['caption2']], mask=mask,
                                                            mask_remove=init_mask_remove,
                                                            mask_add=init_mask_add)
            new_p_img = os.path.join(args.output_dir, k+ '_output.jpg')

            output[0].save(new_p_img)
            if args.save_masks == 1:
                if not os.path.isdir(args.path_mask):
                    os.makedirs(args.path_mask)
                path_mask_final = os.path.join(args.path_mask, k + '.npy')
                with open(path_mask_final, 'wb') as f: np.save(f, final_mask_img)

            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_jar", type=str, default="parser/stanford-parser.jar")
    parser.add_argument("--path_to_models_jar", type=str, default='parser/stanford-parser-4.2.0-models.jar')
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--path_images", type=str)
    parser.add_argument("--path_alignment", type=str)
    parser.add_argument("--path_objects", type=str)
    parser.add_argument("--token_huggingface", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--cache_dir", type=str, default='cache_dir')
    parser.add_argument("--with_objects_sam", type=int, default=1)
    parser.add_argument("--with_diffusion", type=int, default=1)
    parser.add_argument("--with_noise_cancellation", type=int, default=1)
    parser.add_argument("--save_masks", type=int, default=0)
    parser.add_argument("--path_mask", type=str)
    args = parser.parse_args()
    run(args)

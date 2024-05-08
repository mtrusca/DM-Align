
import argparse
import clip, torch, sklearn.preprocessing, collections, pathlib, warnings, tqdm, cv2, json, shutil
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from packaging import version


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    '''
    The text only side for refclipscore
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def compute_clip_score(image_dir, data2):
    # args = parse_args()
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [pathlib.Path(path).stem for path in image_paths]

    candidates = {}
    for obj in data2:
        k = obj['image_filename']
        st, ext = os.path.splitext(obj['image_filename'])
        k = st #+ '_output'
        candidates[k] = obj['caption2']

    # with open(candidates_json) as f:
    #     candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    scores = {image_id: {'CLIPScore': float(clipscore)}
              for image_id, clipscore in
              zip(image_ids, per_instance_image_text)}
    # print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))
    s = []
    for i in scores.keys():
        s.append(scores[i]['CLIPScore'])
    print('clip score', np.mean(s))
    return scores

import numpy as np
from PIL import Image
import os, cv2, json, lpips
from torchvision import transforms

def target_image_caption(data):
  candidates = {}
  for obj in data:
    k = obj['image_filename']
    st, ext = os.path.splitext(obj['image_filename'])
    k = st #+ '_output'
    candidates[k] = obj['caption2']
  return candidates

def pwmse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def compute_pwmse_score(path0, path1):
  files0 = sorted(os.listdir(path0))
  files1 = sorted(os.listdir(path1))
  score_alex1, score_alex2, score_alex3, score_alex4, score_alex5 = [], [], [], [], []
  for i in range(len(files1)):
    img0 = cv2.imread(path0+files0[i])
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.imread(path1+files1[i])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    score_alex1.append(pwmse(img0, img1))
  return score_alex1

def compute_lpips_score(path0, path1, loss_fn_alex):
  files0 = sorted(os.listdir(path0))
  files1 = sorted(os.listdir(path1))
  score_alex1, score_alex2, score_alex3, score_alex4, score_alex5 = [], [], [], [], []
  convert_tensor = transforms.ToTensor()
  for i in tqdm.tqdm(range(len(files1))):
    img0 = Image.open(path0+files0[i])
    img0 = convert_tensor(img0)
    img1 = Image.open(path1+files1[i])
    img1 = convert_tensor(img1)
    score_alex1.append(loss_fn_alex(img0, img1).item())
  return score_alex1

def prepare_image(data, path_images, path_masks, path_images_ref, path_output, back = 0, is_real = 0):
    for i, obj in enumerate(data):
        path_image_ref = os.path.join(path_images_ref, obj['image_filename'])
        shape_ref = cv2.imread(path_image_ref).shape[:2]
        if is_real == 0:
            path_image = os.path.join(path_images,
                                  obj['image_filename'][:-4]+'_output.jpg' )
        if is_real ==1:
            path_image = os.path.join(path_images,
                                  obj['image_filename'] )
        img = cv2.imread(path_image)
        if back == 1:
            path_mask = os.path.join(path_masks,
                                     obj['image_filename'][:-4] + '.npy')
            with open(path_mask, 'rb') as f:
                mask_remove = np.load(f)
            mask1 = np.stack([mask_remove] * 3)
            mask1 = np.transpose(mask1, (1, 2, 0))
            mask1 = mask1.astype('uint8')
            mask1[mask1 == 0] = 255
            mask1[mask1 == 1] = 0
            mask_img = Image.fromarray(mask1).resize((np.array(img).shape[1], np.array(img).shape[0]))
            mask1 = np.array(mask_img)
            mask1 = np.transpose(mask1, (2, 0, 1))
            mask1 = mask1[0]
            masked_img = cv2.bitwise_and(img, img, mask=mask1)
            masked_img = cv2.resize(masked_img, (shape_ref[1], shape_ref[0]))
        else:
            masked_img = cv2.resize(img, (shape_ref[1], shape_ref[0]))
        path_image_red = os.path.join(path_output,
                                      obj['image_filename'])
        cv2.imwrite(path_image_red, masked_img)

def run(args):
    path_data = args.path_data #'data/json_files/100_images.json'
    path_images = args.path_images #'results/dmalign_100_vicha_1/'
    path_masks = args.path_masks# 'data/masks/masks_100/'
    path_images_ref = args.path_images_ref# 'data/100_images/'
    path_output = 'data/test1/'  # predictie
    path_output_ref = 'data/test0/'   # imagini reale
    back = args.back
    with open(path_data, 'r') as myfile: data = myfile.read()
    data = json.loads(data)
    if os.path.isdir(path_output):
        shutil.rmtree(path_output)
    if not os.path.isdir(path_output):
        os.makedirs(path_output)
    if os.path.isdir(path_output_ref):
        shutil.rmtree(path_output_ref)
    if not os.path.isdir(path_output_ref):
        os.makedirs(path_output_ref)
    loss_fn_alex = lpips.LPIPS(net='alex')
    prepare_image(data, path_images, path_masks, path_images_ref, path_output, back = back, is_real = 0)
    if back == 1:
        prepare_image(data, path_images_ref, path_masks, path_images_ref, path_output_ref, back=back, is_real=1)
        lpips_scores = compute_lpips_score(path_output_ref, path_output, loss_fn_alex)
        mse_scores = compute_pwmse_score(path_output_ref, path_output)
    else:
        lpips_scores = compute_lpips_score(path_images_ref, path_output, loss_fn_alex)
        mse_scores = compute_pwmse_score(path_images_ref, path_output)
    print('######## background ##########', args.back)
    print('######## background ##########', args.back)
    print('######## background ##########', args.back)
    print('LPIPS score', np.mean(lpips_scores))
    print('PWMSE score', np.mean(mse_scores))
    if back == 0:
        clip_scores_dict = compute_clip_score(path_output, data)
        clip_scores=[]
        for i, s in enumerate(clip_scores_dict):
          clip_scores.append(clip_scores_dict[s]['CLIPScore'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--path_masks", type=str)
    parser.add_argument("--path_images", type=str)
    parser.add_argument("--path_images_ref", type=str)
    parser.add_argument("--back", type=int)
    args = parser.parse_args()
    run(args)

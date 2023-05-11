import cv2
import numpy as np
from PIL import Image
import torch, logging
logging.disable(logging.WARNING)
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler , StableDiffusionInpaintPipeline

def load_diffusion(token, cache_dir):
  # token ='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'
  vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16, cache_dir=cache_dir).to("cuda")
  unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                              cache_dir=cache_dir,
                                              use_auth_token=token,
                                              subfolder="unet", torch_dtype=torch.float16).to("cuda")
  tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
  text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
  scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
  pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",cache_dir=cache_dir,revision="fp16",torch_dtype=torch.float16, use_auth_token=token).to("cuda")
  return scheduler, text_encoder, tokenizer, unet, vae, pipe

def load_image(p):
    return Image.open(p).convert('RGB')#.resize((512,512))

def load_image_512(p):
    return Image.open(p).convert('RGB').resize((512,512))

def pil_to_latents(vae, image):
    init_image = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist

def latents_to_pil(vae, latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def text_enc(text_encoder, tokenizer, prompts, maxlen=None):
    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to("cuda"))[0].half()

def prompt_2_img(scheduler, text_encoder, tokenizer, unet, vae, prompts, init_img, g=7.5, seed=100, strength=0.3, steps=50, dim=512):
    text = text_enc(text_encoder, tokenizer, prompts)
    uncond = text_enc(text_encoder, tokenizer, [""], text.shape[1])
    emb = torch.cat([uncond, text])
    if seed: torch.manual_seed(seed)
    scheduler.set_timesteps(steps)
    init_latents = pil_to_latents(vae, init_img)
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timesteps)
    with torch.no_grad(): u, t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)
    pred = u + g * (t - u)
    latents = scheduler.step(pred, timesteps, latents).pred_original_sample
    return latents.detach().cpu()

def create_mask_diffusion(scheduler, text_encoder, tokenizer, unet, vae, init_img, rp, qp, n=20, s=0.5):
    diff = {}
    for idx in range(n):
        orig_noise = prompt_2_img(scheduler, text_encoder, tokenizer, unet, vae,
                                           prompts=rp, init_img=init_img, strength=s, seed=100 * idx)[0]
        query_noise = prompt_2_img(scheduler, text_encoder, tokenizer, unet, vae,
                                            prompts=qp, init_img=init_img, strength=s, seed=100 * idx)[0]
        diff[idx] = (np.array(orig_noise) - np.array(query_noise))
    mask = np.zeros_like(diff[0])
    for idx in range(n):
        mask += np.abs(diff[idx])
    mask = mask.mean(0)
    mask = (mask - mask.mean()) / np.std(mask)
    return mask # (mask > 0).astype("uint8")

def improve_mask(mask):
    mask  = cv2.GaussianBlur(mask*255,(3,3),1) > 0
    return mask.astype('uint8')

def reshape_mask(mask, mask1):
  dim10, dim20 = mask.shape
  dim11, dim21 = mask1.shape
  m1 = mask1[::int(dim11/dim10), ::int(dim21/dim20)]
  m1 = m1[:dim10, :dim20]
  return m1

def dmsedit_diffusion(scheduler, text_encoder, tokenizer, unet, vae, pipe, init_img, rp, qp, mask, mask_in, mask_out, g=7.5, seed=100, strength=0.7, steps=20, dim=512):
    if len(mask_in.shape) > 0:
      mask_in = reshape_mask(mask, mask_in)
      mask = np.minimum(1, np.maximum(0, mask.astype('int64') - mask_in.astype('int64')))
    if len(mask_out.shape) > 0:
      mask_out = reshape_mask(mask, mask_out)
      mask = np.maximum(0, np.minimum(1, mask.astype('int64') + mask_out.astype('int64')))
    mask = mask.astype('uint8')
    output = pipe(prompt=qp,
        image=init_img,
        mask_image=Image.fromarray(mask * 255).resize((512, 512)),
        generator=torch.Generator("cuda").manual_seed(100),
        num_inference_steps=50).images
    input_size = np.array(init_img).shape[:2]
    output[0] = output[0].resize([input_size[1], input_size[0]])
    return mask, output

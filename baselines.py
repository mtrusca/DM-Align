from PIL import Image
from FlexIt.src.flexit_editer import FlexitEditer
import ipywidgets as widgets
from io import BytesIO

import time, os, pathlib, json, torch, cv2, requests, PIL, gc, argparse, copy
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from diffusers.utils import load_image

def download_image_instructpix2pix(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def download_image_diffedit(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results_v2")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    model = ['imagic'] # controlnet, instructpix2pix, flexit, diffedit imagic
    if 'diffedit' in model:
        from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler
        from diffusers.utils import load_image
        pipe = StableDiffusionDiffEditPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
                                                               cache_dir="/cw/liir_code/NoCsBack/maria/diffedit/cache_clip").to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        del StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler
        path_bson_data = 'imagen_v2.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_360.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "diffedit_360/"):
            os.makedirs("results/" + args.path + "diffedit_360/")
        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            img_ = data2[i]['image_filename'].replace('_', '-')
            url = [img for img in images if img_ in img][0].strip()
            target_prompt = data2[i]['caption2']
            source_prompt = data2[i]['caption1']
            init_image = load_image(url).resize((768, 768))
            torch.cuda.empty_cache()
            gc.collect()
            mask_image = pipe.generate_mask(image=init_image, source_prompt=source_prompt, target_prompt=target_prompt)
            torch.cuda.empty_cache()
            gc.collect()
            image_latents = pipe.invert(image=init_image, prompt=source_prompt).latents
            torch.cuda.empty_cache()
            gc.collect()
            image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]
            torch.cuda.empty_cache()
            gc.collect()
            path_output = os.path.join("/cw/liir_code/NoCsBack/maria/diffedit/results/" + args.path + "diffedit_360",
                                       data2[i]['image_filename'][:-4] + '_output.jpg')
            img1 = os.path.join('data/imagen', data2[i]['image_filename'])
            s = np.array(Image.open(img1)).shape[:2]
            image = image.resize([s[1], s[0]])
            image.save(path_output)

        #################################
        #################################
        #################################
        path_bson_data = '100_images.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_100.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "diffedit_100/"):
            os.makedirs("results/" + args.path + "diffedit_100/")
        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            img_ = data2[i]['image_filename'].replace('_', '-')
            url = [img for img in images if img_ in img][0].strip()
            target_prompt = data2[i]['caption2']
            source_prompt = data2[i]['caption1']
            init_image = load_image(url).resize((768, 768))
            torch.cuda.empty_cache()
            gc.collect()
            mask_image = pipe.generate_mask(image=init_image, source_prompt=source_prompt, target_prompt=target_prompt)
            torch.cuda.empty_cache()
            gc.collect()
            image_latents = pipe.invert(image=init_image, prompt=source_prompt).latents
            torch.cuda.empty_cache()
            gc.collect()
            image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]
            torch.cuda.empty_cache()
            gc.collect()
            path_output = os.path.join("/cw/liir_code/NoCsBack/maria/diffedit/results/" + args.path + "diffedit_100",
                                       data2[i]['image_filename'][:-4] + '_output.jpg')
            img1 = os.path.join('data/100_images', data2[i]['image_filename'])
            s = np.array(Image.open(img1)).shape[:2]
            image = image.resize([s[1], s[0]])
            image.save(path_output)

    if 'instructpix2pix' in model:
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None,
                                                                      cache_dir="/cw/liir_code/NoCsBack/maria/diffedit/cache_clip").to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        path_bson_data = 'imagen_v2.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_360.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "/instructpix2pix_360/"):
            os.makedirs("results/"+ args.path + "/instructpix2pix_360/")
        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            start = time.time()
            img_ = data2[i]['image_filename'].replace('_', '-')
            url = [img for img in images if img_ in img][0].strip()
            try:
                image = download_image_instructpix2pix(url).resize((512, 512))
            except:
                try:
                    image = download_image_instructpix2pix(url).resize((512, 512))
                except:
                    try:
                        image = download_image_instructpix2pix(url).resize((512, 512))
                    except:
                        img_init = os.path.join('data/imagen', data2[i]['image_filename'])
                        image = np.array(Image.open(img_init)).shape[:2]

            prompt = data2[i]['caption2']
            torch.cuda.empty_cache()
            gc.collect()
            image = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1).images[0]
            torch.cuda.empty_cache()
            gc.collect()
            path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/instructpix2pix_360',
                                       data2[i]['image_filename'][:-4] + '_output.jpg')
            img1 = os.path.join('data/imagen', data2[i]['image_filename'])
            s = np.array(Image.open(img1)).shape[:2]
            image = image.resize([s[1], s[0]])
            image.save(path_output)
            end = time.time() - start
            # print('time  1', end)

        ################################
        ################################
        ################################
        path_bson_data = '100_images.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_100.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "/instructpix2pix_100/"):
            os.makedirs("results/"+ args.path + "/instructpix2pix_100/")
        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            start = time.time()
            img_ = data2[i]['image_filename'].replace('_', '-')
            url = [img for img in images if img_ in img][0].strip()
            try:
                image = download_image_instructpix2pix(url).resize((512, 512))
            except:
                try:
                    image = download_image_instructpix2pix(url).resize((512, 512))
                except:
                    try:
                        image = download_image_instructpix2pix(url).resize((512, 512))
                    except:
                        img_init = os.path.join('data/100_images', data2[i]['image_filename'])
                        image = np.array(Image.open(img_init)).shape[:2]
            prompt = data2[i]['caption2']
            torch.cuda.empty_cache()
            gc.collect()
            image = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1).images[0]
            torch.cuda.empty_cache()
            gc.collect()
            path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/instructpix2pix_100',
                                       data2[i]['image_filename'][:-4] + '_output.jpg')
            img1 = os.path.join('data/100_images', data2[i]['image_filename'])
            s = np.array(Image.open(img1)).shape[:2]
            image = image.resize([s[1], s[0]])
            image.save(path_output)
            end = time.time() - start
            # print('time  2', end)

        ################################
        ################################
        ################################
        path_bson_data = 'bison_dataset.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_1437.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "/instructpix2pix_1437/"):
            os.makedirs("results/"+ args.path + "/instructpix2pix_1437/")
        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            start = time.time()
            img_ = data2[i]['image_filename'].replace('_', '-')
            img_ = img_.replace('--', '-')
            url = [img for img in images if img_ in img][0].strip()
            try:
                image = download_image_instructpix2pix(url).resize((512, 512))
            except:
                try:
                    image = download_image_instructpix2pix(url).resize((512, 512))
                except:
                    try:
                        image = download_image_instructpix2pix(url).resize((512, 512))
                    except:
                        img_init = os.path.join('bison_data', data2[i]['image_filename'])
                        image = np.array(Image.open(img_init)).shape[:2]
            prompt = data2[i]['caption2']
            torch.cuda.empty_cache()
            gc.collect()
            image = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1).images[0]
            torch.cuda.empty_cache()
            gc.collect()
            path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/instructpix2pix_1437',
                                       data2[i]['image_filename'][:-4] + '_output.jpg')
            img1 = os.path.join('bison_data', data2[i]['image_filename'])
            s = np.array(Image.open(img1)).shape[:2]
            image = image.resize([s[1], s[0]])
            image.save(path_output)
            end = time.time() - start
            # print('time  2', end)

    if 'controlnet' in model:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,  DiffusionPipeline, DDIMScheduler
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16,
                                                     cache_dir="/cw/liir_code/NoCsBack/maria/diffedit/cache_clip")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", controlnet=controlnet, torch_dtype=torch.float16,
            cache_dir="/cw/liir_code/NoCsBack/maria/diffedit/cache_clip")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        generator = torch.manual_seed(args.seed)

        path_bson_data = 'imagen_v2.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_360.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "/controlnet_360/"):
            os.makedirs("results/" + args.path + "/controlnet_360/")

        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            img_ = data2[i]['image_filename'].replace('_', '-')
            # img_ = img_.replace('--', '-')
            path = [img for img in images if img_ in img]
            if len(path) > 0:
                path = path[0]
                image = load_image(path).resize((512, 512))
                image = np.array(image)
                image = cv2.Canny(image, 100, 100)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                canny_image = Image.fromarray(image)
                image = pipe(data2[i]['caption2'], num_inference_steps=50, generator=generator, image=canny_image).images[0]
                path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/controlnet_360',
                                           data2[i]['image_filename'][:-4]+'_output.jpg')
                img1 = os.path.join('data/imagen', data2[i]['image_filename'])
                s = np.array(Image.open(img1)).shape[:2]
                image = image.resize([s[1], s[0]])
                image.save(path_output)
            if len(path) == 0:
                path = os.path.join('data/imagen', data2[i]['image_filename'])
                path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/controlnet_360',
                                           data2[i]['image_filename'][:-4]+'_output.jpg')
                image = Image.open(path)
                image.save(path_output)

        #################################
        #################################
        #################################
        path_bson_data = '100_images.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_100.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "/controlnet_100/"):
            os.makedirs("results/" + args.path + "/controlnet_100/")

        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            img_ = data2[i]['image_filename'].replace('_', '-')
            # img_ = img_.replace('--', '-')
            path = [img for img in images if img_ in img]
            if len(path) > 0:
                path = path[0]
                image = load_image(path).resize((512, 512))
                image = np.array(image)
                image = cv2.Canny(image, 100, 100)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                canny_image = Image.fromarray(image)
                image = pipe(data2[i]['caption2'], num_inference_steps=50, generator=generator, image=canny_image).images[0]
                path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/controlnet_100',
                                           data2[i]['image_filename'][:-4]+'_output.jpg')
                img1 = os.path.join('data/100_images', data2[i]['image_filename'])
                s = np.array(Image.open(img1)).shape[:2]
                image = image.resize([s[1], s[0]])
                image.save(path_output)
            if len(path) == 0:
                path = os.path.join('data/100_imagess', data2[i]['image_filename'])
                path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/controlnet_100',
                                           data2[i]['image_filename'][:-4]+'_output.jpg')
                image = Image.open(path)
                image.save(path_output)

        #################################
        #################################
        #################################
        path_bson_data = 'bison_dataset.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_1437.txt")
        images = a.readlines()
        if not os.path.isdir("results/" + args.path + "/controlnet_1437/"):
            os.makedirs("results/" + args.path + "/controlnet_1437/")

        for i in tqdm(range(len(data2))):
            print('imaginea', i)
            img_ = data2[i]['image_filename'].replace('_', '-')
            img_ = img_.replace('--', '-')
            # img_ = img_.replace('--', '-')
            path = [img for img in images if img_ in img]
            if len(path) > 0:
                path = path[0]
                image = load_image(path).resize((512, 512))
                image = np.array(image)
                image = cv2.Canny(image, 100, 100)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                canny_image = Image.fromarray(image)
                image = pipe(data2[i]['caption2'], num_inference_steps=50, generator=generator, image=canny_image).images[0]
                path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/controlnet_1437',
                                           data2[i]['image_filename'][:-4]+'_output.jpg')
                img1 = os.path.join('bison_data', data2[i]['image_filename'])
                s = np.array(Image.open(img1)).shape[:2]
                image = image.resize([s[1], s[0]])
                image.save(path_output)
            if len(path) == 0:
                path = os.path.join('bison_data', data2[i]['image_filename'])
                path_output = os.path.join('/cw/liir_code/NoCsBack/maria/diffedit/results/' + args.path + '/controlnet_1437',
                                           data2[i]['image_filename'][:-4]+'_output.jpg')
                image = Image.open(path)
                image.save(path_output)

    if 'flexit' in model:
        transformer = FlexitEditer(
            names = ['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16'],
        img_size = 288,
        mode = 'vqgan',
        lr=0.05,
        device = 'cuda:0',
        n_augment = 1,
        znorm_weight = 0.05,
        im0_coef = 0.2,
        source_coef=0.4,
        latent_space = 'vqgan',
        lpips_weight = 0.15,
        )
        path_bson_data = 'imagen_v2.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        if not os.path.isdir("results/"+ args.path + "/flexit_360/"):
            os.makedirs("results/" + args.path + "/flexit_360")

        for i, obj in tqdm(enumerate(data2)):
            start = time.time()
            p = 'data/imagen/' + obj['image_filename']
            k = pathlib.Path(obj['image_filename']).stem
            new_p = 'imagen/' + k + '_output.jpg'
            if os.path.isfile(p) and not os.path.isfile(new_p):
                img0 = Image.open(p).convert('RGB')
                bio = BytesIO()
                img0.save(bio, format='png')
                im = widgets.Image(
                    value=bio.getvalue(),
                    format='png',
                    width=384,
                    height=384)
                transformer.args.max_iter = 160
                out_imgs, _ = transformer(img0,
                                           obj['caption1'],
                                           obj['caption2'])
                out_img = out_imgs[transformer.args.max_iter]
                print('image size', np.array(out_img).shape)
                dim1, dim2 = np.array(img0).shape[:2]
                out_img = out_img.resize((dim2, dim1))
                k = pathlib.Path(obj['image_filename']).stem
                new_p = 'results/' + args.path + '/flexit_360/' + k + '_output.jpg'
                out_img.save(new_p)
                end = time.time() - start
                # print('time', end)


        path_bson_data = '100_images.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)

        if not os.path.isdir("results/" +args.path + "/flexit_100/"):
            os.makedirs("results/" + args.path + "/flexit_100/")

        for i, obj in tqdm(enumerate(data2)):
            start = time.time()
            p = '100_images/' + obj['image_filename']
            k = pathlib.Path(obj['image_filename']).stem
            new_p = '100_images/' + k + '_output.jpg'
            if os.path.isfile(p) and not os.path.isfile(new_p):
                img0 = Image.open(p).convert('RGB')
                bio = BytesIO()
                img0.save(bio, format='png')
                im = widgets.Image(
                    value=bio.getvalue(),
                    format='png',
                    width=384,
                    height=384)
                transformer.args.max_iter = 160
                out_imgs, _ = transformer(img0,
                                           obj['caption1'],
                                           obj['caption2'])
                out_img = out_imgs[transformer.args.max_iter]
                print('image size', np.array(out_img).shape)
                dim1, dim2 = np.array(img0).shape[:2]
                out_img = out_img.resize((dim2, dim1))
                k = pathlib.Path(obj['image_filename']).stem
                new_p = 'results/' + args.path + '/flexit_100/' + k + '_output.jpg'
                out_img.save(new_p)
                end = time.time() - start
                # print('time', end)

        path_bson_data = 'bison_dataset.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)

        if not os.path.isdir("results/" +args.path + "/flexit_1437/"):
            os.makedirs("results/" + args.path + "/flexit_1437/")

        for i, obj in tqdm(enumerate(data2)):
            start = time.time()
            p = 'bison_data/' + obj['image_filename']
            k = pathlib.Path(obj['image_filename']).stem
            new_p = 'bison_data/' + k + '_output.jpg'
            if os.path.isfile(p) and not os.path.isfile(new_p):
                img0 = Image.open(p).convert('RGB')
                bio = BytesIO()
                img0.save(bio, format='png')
                im = widgets.Image(
                    value=bio.getvalue(),
                    format='png',
                    width=384,
                    height=384)
                transformer.args.max_iter = 160
                out_imgs, _ = transformer(img0,
                                           obj['caption1'],
                                           obj['caption2'])
                out_img = out_imgs[transformer.args.max_iter]
                print('image size', np.array(out_img).shape)
                dim1, dim2 = np.array(img0).shape[:2]
                out_img = out_img.resize((dim2, dim1))
                k = pathlib.Path(obj['image_filename']).stem
                new_p = 'results/' + args.path + '/flexit_1437/' + k + '_output.jpg'
                out_img.save(new_p)
                end = time.time() - start
                # print('time', end)
    #
    if 'imagic' in model:
        torch_dtype = torch.float32
        from diffusers import DDIMScheduler, DDIMInverseScheduler
        from pipelines.imagic_pipeline import ImagicStableDiffusionPipeline

        generator = torch.manual_seed(args.seed)

        device = 'cuda'
        path_bson_data = '/cw/liir_code/NoCsBack/maria/diffedit/data/json_files/100_images.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        if not os.path.isdir("/cw/liir_code/NoCsBack/maria/diffedit/results/"+ args.path + "/imagic_100/"):
            os.makedirs("/cw/liir_code/NoCsBack/maria/diffedit/results/" + args.path + "/imagic_100")

        for i in tqdm(range(len(data2))):

            start = time.time()
            pipe = ImagicStableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                safety_checker=None,
                custom_pipeline="imagic_stable_diffusion",
                cache_dir = 'cache_dir',
                scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        clip_sample=False, set_alpha_to_one=False))
            pipe.to("cuda")
            print('imaginea', i)
            raw_image = Image.open('/cw/liir_code/NoCsBack/maria/diffedit/data/100_images/' + data2[i]['image_filename']).convert("RGB").resize((512, 512))
            caption = data2[i]['caption2']
            res = pipe.train(
                caption,
                image=raw_image,
                generator=generator,
                text_embedding_optimization_steps=100,
                model_fine_tuning_optimization_steps=100, )

            res = pipe(alpha=1, guidance_scale=7.5, num_inference_steps=50)
            image = res.images[0]
            new_p = "/cw/liir_code/NoCsBack/maria/diffedit/results/" + args.path + "/imagic_100" + data2[i]['image_filename'][:-4] + '_output.jpg'
            image.save(new_p)
            del pipe
            end = time.time()
            print('end', end)

        path_bson_data = '/cw/liir_code/NoCsBack/maria/diffedit/data/json_files/imagen_v2.json'
        with open(path_bson_data, 'r') as myfile: data2=myfile.read()
        data2 = json.loads(data2)
        # a = open("/cw/liir_code/NoCsBack/maria/diffedit/files_postimage/files_360.txt")
        # images = a.readlines()
        if not os.path.isdir("/cw/liir_code/NoCsBack/maria/diffedit/results/"+ args.path + "/imagic_360/"):
            os.makedirs("/cw/liir_code/NoCsBack/maria/diffedit/results/" + args.path + "/imagic_360")

        for i in tqdm(range(len(data2))):
            start = time.time()
            pipe = ImagicStableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                safety_checker=None,
                cache_dir='cache_dir',
                custom_pipeline="imagic_stable_diffusion",
                scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        clip_sample=False, set_alpha_to_one=False))
            pipe.to("cuda")
            print('imaginea', i)
            raw_image = Image.open('/cw/liir_code/NoCsBack/maria/diffedit/data/imagen/' + data2[i]['image_filename']).convert("RGB").resize((512, 512))
            caption = data2[i]['caption2']
            res = pipe.train(
                caption,
                image=raw_image,
                generator=generator,
                text_embedding_optimization_steps=100,
                model_fine_tuning_optimization_steps=100, )

            res = pipe(alpha=1, guidance_scale=7.5, num_inference_steps=50)
            image = res.images[0]
            new_p = "/cw/liir_code/NoCsBack/maria/diffedit/results/" + args.path + "/imagic_360" + data2[i]['image_filename'][:-4] + '_output.jpg'
            image.save(new_p)
            del pipe
            end = time.time() - start
            print('end', end)

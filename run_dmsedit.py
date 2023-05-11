
import logging
import argparse
from PIL import Image
logging.disable(logging.WARNING)
from nltk.stem import WordNetLemmatizer
import nltk, os, json, ast, pathlib, string
from nltk.parse.stanford import StanfordDependencyParser
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

from detr import load_detr
from vicha_gradcam import load_vicha
from diffusion import load_diffusion
from dmsedit import dmsedit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_input_data", type = str)
    parser.add_argument("--path_source_images", type = str)
    parser.add_argument("--path_alignments", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--bert_config_path", type = str)
    parser.add_argument("--path_to_jar", type = str)
    parser.add_argument("--path_to_models_jar", type = str)
    parser.add_argument("--path_target_images", type = str)
    parser.add_argument("--token", type = str)
    parser.add_argument("--cache_dir", type =  str)
    args = parser.parse_args()

    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)

    if not os.path.isdir(args.path_target_images):
        os.makedirs(args.path_target_images)

    with open(args.path_input_data, 'r') as myfile: data=myfile.read()
    data = json.loads(data)
    with open(args.path_alignments, 'r') as myfile: alignments=myfile.read()
    alignments = json.loads(alignments)

    scheduler, text_encoder, tokenizer, unet, vae, pipe_diffusion = load_diffusion(args.token, args.cache_dir)
    vicha_tokenizer, vicha_model = load_vicha(args.model_path, args.bert_config_path)
    feature_extractor_detr, model_detr = load_detr()
    dependency_parser = StanfordDependencyParser(path_to_jar=args.path_to_jar, path_to_models_jar=args.path_to_models_jar)
    wordnet_lemmatizer = WordNetLemmatizer()

    for i, obj in enumerate(data):
        path_image = args.path_source_images + obj['image_filename']
        obj['caption1'] = obj['caption1'].strip().lower()
        obj['caption2']=obj['caption2'].strip().lower()
        obj['caption1'] = obj['caption1'].translate(str.maketrans('', '', string.punctuation))
        obj['caption2'] = obj['caption2'].translate(str.maketrans('', '', string.punctuation))
        alignment = ast.literal_eval(alignments[obj['image_filename']])
        target_image = dmsedit(path_image, obj['caption1'], obj['caption2'], alignment,
                scheduler, text_encoder, tokenizer, unet, vae, pipe_diffusion,
                vicha_tokenizer, vicha_model,
                feature_extractor_detr, model_detr,
                dependency_parser, wordnet_lemmatizer)
        k = pathlib.Path(obj['image_filename']).stem
        new_path_image = args.path_target_images + k + '_output.jpg'
        target_image.save(new_path_image)
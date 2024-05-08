import numpy as np
import gc, os
import string, pathlib, argparse
import ast, json, torch
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem import WordNetLemmatizer
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

import spacy
import nltk
nltk.download('punkt', download_dir='testve/nltk_data')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger', download_dir='testve/nltk_data')
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet', download_dir='testve/nltk_data')
nltk.download('omw-1.4', download_dir='testve/nltk_data')

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
    alignment_words = []
    add1, remove1, add2 = [], [], []
    # inds2 = []
    # for pair in list(alignment[0]):
    #   _, ind2 = [int(i) for i in pair.split('-')]
    #   inds2.append(ind2)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_stanford_parser", default='parser/stanford-parser.jar', type=str)
    parser.add_argument("--path_to_stanford_parser_models", default='parser/stanford-parser-4.2.0-models.jar', type=str)
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--path_alignment", type=str)
    parser.add_argument("--path_images", type=str)
    parser.add_argument("--path_objects", type=str)
    args = parser.parse_args()
    with open(args.path_data, 'r') as myfile:
        data = myfile.read()
    data = json.loads(data)
    with open(args.path_alignment, 'r') as myfile:
        alignments = myfile.read()
    alignments = json.loads(alignments)
    dependency_parser = StanfordDependencyParser(path_to_jar=args.path_to_stanford_parser, path_to_models_jar=args.path_to_stanford_parser_models)
    wordnet_lemmatizer = WordNetLemmatizer()

    dict_obj = {}
    for i, obj in enumerate(data):
        print('imaginea ', i)
        p = os.path.join(args.path_images , obj['image_filename'])
        obj['caption1'] = obj['caption1'].strip().lower()  # .replace('.','').replace("'s", 's')
        obj['caption2'] = obj['caption2'].strip().lower()  # .replace('.','').replace("'s", 's')
        obj['caption1'] = obj['caption1'].translate(str.maketrans('', '', string.punctuation))
        obj['caption2'] = obj['caption2'].translate(str.maketrans('', '', string.punctuation))
        k = pathlib.Path(obj['image_filename']).stem
        print(k)
        print('p', p)
        if os.path.isfile(p):
            alignment = ast.literal_eval(alignments[obj['image_filename']])
            noun_modifiers_pairs1, noun_modifiers_pairs2, noun_verb_pairs = detect_noun_relation(obj['caption1'],
                                                                                                 obj['caption2'],
                                                                                                 dependency_parser)
            add1, add2, remove1, add1_tok, add2_tok, remove1_tok = extract_words_alignment(obj['caption1'], obj['caption2'],
                                                                                           alignment,
                                                                                           noun_modifiers_pairs1,
                                                                                           noun_modifiers_pairs2,
                                                                                           noun_verb_pairs,
                                                                                           wordnet_lemmatizer)
        temp1, temp2 = [], []
        add1_tok = [i for i in add1_tok if i != 'photo']
        add1_tok = [i for i in add1_tok if i != 'painting']
        remove1_tok = [i for i in remove1_tok if i != 'photo']
        remove1_tok = [i for i in remove1_tok if i != 'painting']
        remove1_tok_dict = {}
        for i in remove1_tok: remove1_tok_dict[i] = i
        add1_tok_dict = {}
        for i in add1_tok: add1_tok_dict[i] = i
        if add1_tok_dict != {}:
            base_model1 = GroundedSAM(ontology=CaptionOntology(add1_tok_dict))
            results1 = base_model1.predict(p)
            maps1 = tuple([m for m in results1.mask])
            maps1 = np.logical_or.reduce((maps1))
            path = os.path.join(args.path_objects , k + '_add.npy')
            with open(path, 'wb') as f:
                np.save(f, maps1)
            temp1.append(maps1)
        torch.cuda.empty_cache()
        gc.collect()
        if remove1_tok_dict != {}:
            base_model2 = GroundedSAM(ontology=CaptionOntology(remove1_tok_dict))
            results2 = base_model2.predict(p)
            maps2 = tuple([m for m in results2.mask])
            maps2 = np.logical_or.reduce((maps2))
            path = os.path.join(args.path_objects , k + '_remove.npy')
            with open(path, 'wb') as f:
                np.save(f, maps2)
            temp2.append(maps2)


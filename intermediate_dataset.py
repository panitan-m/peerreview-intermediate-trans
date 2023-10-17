import logging
import numpy as np
from ASAP.asap import ASAP
from datasets import Dataset
from collections import Counter
from ReviewAdvisor.tagger.helper.split import get_sents
from tqdm import tqdm

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ASPECTS = [
    'clarity',
    'meaningful_comparison',
    'motivation',
    'originality',
    'replicability',
    'soundness',
    'substance']

ASPECTS_ABBR = [
    'CLR',
    'COM',
    'MOT',
    'ORI',
    'REP',
    'SOU',
    'SUB'
]

def process_content(content, extract, split):
    content = content.encode('utf-8','ignore').decode("utf-8")
    if extract:
        raise NotImplementedError
        content = extractor.extract(content)
    if split:
        content = get_sents(content)
    return content

def process_decision(decision):
    if 'reject' in decision.lower():
        return 0
    else:
        return 1

def data_to_dict(dataset, extract=False, split=False):
    data = {
        "paper": [],
        "review": [],
        "label": [],
        "decision": [],
        "score": [],
    }

    for d in tqdm(dataset):
        paper_content = d.get_paper_content()
        if paper_content is not None:
            paper_content = process_content(paper_content, extract, split)
            review_content = d.get_review_content()
            review_content = process_content(review_content, extract, split)
            aspect_labels = []
            for review in d.REVIEWS:
                for _, _, label in review['labels']:
                    if 'summary' not in label:
                        aspect_labels.append(label)
            scores = []
            for score in d.SCORE:
                if 'rating' in score.keys():
                    scores.append(int(score['rating'][0]))
            if len(aspect_labels) > 0: 
                if (not split) or (split and len(paper_content) > 30):
                    data['paper'].append(paper_content)
                    data['review'].append(review_content)
                    data['label'].append(Counter(aspect_labels))
                    data['decision'].append(process_decision(d.DECISION))
                    if len(scores) > 0:
                        data['score'].append((np.average(scores)-1)/9)
                    else:
                        data['score'].append(None)
    return data


def polarity_cls(counter, aspect, num_classes):
    pos_key = '_'.join([aspect, 'positive'])
    neg_key = '_'.join([aspect, 'negative'])
    if num_classes == 1:
        if counter[pos_key] == 0 and counter[neg_key] == 0:
            return -1
        elif counter[neg_key] == 0:
            return 0
        else:
            return counter[pos_key] / (counter[pos_key] + counter[neg_key])
    if counter[pos_key] > counter[neg_key]:
        if num_classes == 3:
            return 2
        else: 
            return 1
    elif counter[pos_key] < counter[neg_key]:
        return 0
    else:
        if counter[pos_key] != 0 and num_classes == 3:
            return 1
        return -1

def label_preprocess(batch, aspect, num_classes, check_score=False):
    if aspect == 'decision':
        batch['label'] = batch['decision']
        return batch
    if aspect == 'score':
        batch['label'] = batch['score']
        del_indices = [i for i, v in enumerate(batch['label']) if v is None]
        for i in sorted(del_indices, reverse=True):
            for key in batch.keys():
                del batch[key][i]
        return batch
    label = batch['label']
    score = batch['score']
    decision = batch['decision']
    del_indices = []
    for i, (l, d, s) in enumerate(zip(label, decision, score)):
        l = {k: 0 if v is None else v for k, v in l.items()}
        if aspect is not None:
            new_label = polarity_cls(l, aspect, num_classes)
        else:
            new_label = []
            for _aspect in ASPECTS:
                new_label.append(polarity_cls(l, _aspect, num_classes))
        label[i] = new_label
        if check_score and num_classes == 2:
            if (new_label == 1 and s <= 0.45) or (new_label == 0 and s > 0.45):
            # if (new_label == 1 and d == 0) or (new_label == 0 and d == 1):
                new_label = -1
            
        if (np.array(new_label) == -1).all():
            del_indices.append(i)
    for i in sorted(del_indices, reverse=True):
        for key in batch.keys():
            del batch[key][i]
    return batch


def batch_tokenize_preprocess(batch, tokenizer, max_length):
    paper, review, labels = batch["paper"], batch["review"], batch["label"]
    paper_text_tokenized = tokenizer(
        paper, padding="max_length", truncation=True, max_length=max_length
    )
    review_text_tokenized = tokenizer(
        review, padding="max_length", truncation=True, max_length=max_length
    )
    batch = {"paper_"+k: v for k, v in paper_text_tokenized.items()}
    for k, v in review_text_tokenized.items():
        batch["review_"+k] = v
    if isinstance(labels[0], list):
        batch["labels"] = [
            [-100 if score == -1 else score for score in l]
            for l in labels
        ]
    else:
        batch['labels'] = labels
    return batch


def sentence_encode(sents, model, max_length):
    padding = [''] * (max_length - len(sents))
    padded = sents + padding
    padded = padded[:max_length]
    sents_encoded = model.encode(padded, show_progress_bar=False, batch_size=128)
    mask = [1] * len(sents)
    mask += [0] * (max_length - len(sents))
    mask = np.array(mask[:max_length])
    return sents_encoded, mask

def batch_encode_preprocess(batch, model, max_length):
    paper, review, labels = batch["paper"], batch["review"], batch["label"]
    paper_sent_encoded = list(zip(*list(map(lambda x: sentence_encode(x, model, max_length), paper))))
    # review_sent_encoded = list(zip(*list(map(lambda x: sentence_encode(x, model, max_length), review))))

    batch["paper_sentence_embeddings"] = list(paper_sent_encoded[0])
    batch["paper_sentence_masks"] = list(paper_sent_encoded[1])
    # batch["review_sentence_embeddings"] = list(review_sent_encoded[0])
    # batch["review_sentence_masks"] = list(review_sent_encoded[1])
    if isinstance(labels[0], list):
        batch["labels"] = [[-100 if score == -1 else score for score in l] for l in labels]
    else:
        batch['labels'] = labels
    return batch


def get_dataset(
    con_name, 
    pretrained, 
    extract=False, 
    split=False, 
    max_length=512, 
    aspect=None, 
    num_classes=3,
    check_score=False,
    seed=1234):
    
    if aspect is not None:
        aspect = aspect.lower()
        if aspect != 'decision' and aspect != 'score':
            assert aspect in ASPECTS
    
    dataset = ASAP(con_name)
    dataset = data_to_dict(dataset, extract, split)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(
        lambda batch: label_preprocess(batch, aspect, num_classes, check_score),
        batched=True,
    )
    
    train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed).values()
    val_dataset, test_dataset = test_dataset.train_test_split(test_size=0.5, shuffle=True, seed=seed).values()
    
    logger.info("Train: {}".format(len(train_dataset)))
    logger.info("Val: {}".format(len(val_dataset)))
    logger.info("Test: {}".format(len(test_dataset)))
    
    test_label = np.array(test_dataset['label'])
    if aspect is not None:
        test_majority = 0
        for s in list(set(test_label)):
            s_ratio = (test_label == s).sum()/len(test_dataset)
            if s_ratio > test_majority: 
                test_majority = s_ratio
        logger.info("Test majority: {:.3f}".format(test_majority))
    else:
        test_majority = [max(v, 1 - v) for v in (test_label == 1).sum(0)/(test_label != -1).sum(0)]
        logger.info("Test majority:")
        info = ""
        for asp, majority in zip(ASPECTS_ABBR, test_majority):
            info += "{}: {:.3f}   ".format(asp, majority)
        logger.info(info)
        
    logger.info("Tokenizing")
    if split:
        tool = SentenceTransformer(pretrained)
        batch_preprocess = batch_encode_preprocess
    else:
        tool = AutoTokenizer.from_pretrained(pretrained)
        batch_preprocess = batch_tokenize_preprocess
        
    train_dataset = train_dataset.map(
        lambda batch: batch_preprocess(batch, tool, max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda batch: batch_preprocess(batch, tool, max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    test_dataset = test_dataset.map(
        lambda batch: batch_preprocess(batch, tool, max_length),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    test_dataset.set_format(type='torch')
    
    return train_dataset, val_dataset, test_dataset


def get_train_val(
    con_name, 
    pretrained, 
    extract=False, 
    split=False, 
    max_length=512, 
    aspect=None, 
    num_classes=3, 
    check_score=False,
    seed=1234):
    
    if aspect is not None:
        aspect = aspect.lower()
        if aspect != 'decision' and aspect != 'score':
            assert aspect in ASPECTS
    
    dataset = ASAP(con_name)
    dataset = data_to_dict(dataset, extract, split)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(
        lambda batch: label_preprocess(batch, aspect, num_classes, check_score),
        batched=True,
    )
    
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed).values()
    
    # train_dataset = Dataset.from_dict(train_dataset[:500])
    
    logger.info("Train: {}".format(len(train_dataset)))
    logger.info("Val: {}".format(len(val_dataset)))
    
    test_label = np.array(val_dataset['label'])
    if aspect is not None:
        test_majority = 0
        for s in list(set(test_label)):
            s_ratio = (test_label == s).sum()/len(val_dataset)
            if s_ratio > test_majority: 
                test_majority = s_ratio
        logger.info("Test majority: {:.3f}".format(test_majority))
    else:
        test_majority = [max(v, 1 - v) for v in (test_label == 1).sum(0)/(test_label != -1).sum(0)]
        logger.info("Test majority:")
        info = ""
        for asp, majority in zip(ASPECTS_ABBR, test_majority):
            info += "{}: {:.3f}   ".format(asp, majority)
        logger.info(info)
        
    logger.info("Tokenizing")
    if split:
        tool = SentenceTransformer(pretrained)
        batch_preprocess = batch_encode_preprocess
    else:
        tool = AutoTokenizer.from_pretrained(pretrained)
        batch_preprocess = batch_tokenize_preprocess
        
    train_dataset = train_dataset.map(
        lambda batch: batch_preprocess(batch, tool, max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda batch: batch_preprocess(batch, tool, max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    
    return train_dataset, val_dataset
    
if __name__ == '__main__':
    pretrained = 'allenai/scibert_scivocab_uncased'
    get_dataset(pretrained, content='review')
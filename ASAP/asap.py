import json
from .PaperReader import PaperReader


class ASAP:
    def __init__(self, con_name):
        file_path = '../ReviewAdvisor/dataset/aspect_data/review_with_aspect.jsonl'
        paper_ids = []
        reviews = {}
        with open(file_path, 'r') as f:
            for json_str in f:
                aspect_data = json.loads(json_str)
                paper_id = aspect_data['id']
                if con_name.lower() in paper_id.lower():
                    if paper_id in reviews:
                        reviews[paper_id].append(aspect_data)
                    else:
                        reviews[paper_id] = [aspect_data]
                    paper_ids.append(paper_id)
        paper_ids = sorted(list(set(paper_ids)))
        self.data = []
        for paper_id in paper_ids:
            paper = PaperReader.from_id(paper_id)
            paper.REVIEWS = reviews[paper_id]
            self.data.append(paper)
            
    def __len__(self):
        return len(self.data)
            
    def __iter__(self):
        return (d for d in self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
class AspectDataset():
    def __init__(
        self, 
        tokenizer=None, 
        split_sentence=False,
        paper_max_length=1024,
        aspect_max_length=256,
        ):
        
        self.tokenizer = tokenizer
        self.paper_max_length = paper_max_length
        self.aspect_max_length = aspect_max_length
        
        self.x = []
        self.y = []
        self.a = []
        asap_dataset = ASAP()
        for d in asap_dataset:
            content = d.get_paper_content()
            if content is not None: 
                for review in d.REVIEWS:
                    for start, end, label in review['labels']:
                        self.x.append(content)
                        self.y.append(review['text'][start:end])
                        self.a.append(label)
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.tokenizer is not None:
            paper = self.tokenizer(self.x[index], padding='max_length', truncation=True, max_length=self.paper_max_length, return_tensors='pt')
            aspect = self.tokenizer(self.y[index], padding='max_length', truncation=True, max_length=self.aspect_max_length, return_tensors='pt')
            
            return {
                'input_ids': paper['input_ids'].squeeze(0),
                'attention_mask': paper['attention_mask'].squeeze(0),
                'decoder_input_ids': aspect['input_ids'].squeeze(0),
                'decoder_attention_mask': aspect['attention_mask'].squeeze(0)
            }
        else:
            return {
                'paper': self.x[index],
                'aspect': self.y[index],
                'label': self.a[index]
            }

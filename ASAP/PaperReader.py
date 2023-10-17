import os
import json

from .Paper import Paper
    
    
class PaperReader:
    
    @staticmethod
    def from_id(paper_id):
        con, year, id = paper_id.split('_')
        folder_name = con + '_' + year
        file_path = os.path.join('ReviewAdvisor', 'dataset', folder_name, folder_name+'_content')
        file_name = paper_id+'_content.json'
        
        with open(os.path.join(file_path, file_name), 'r') as f:
            data = json.load(f)
            
        metadata = data['metadata']
        title = metadata['title']
        abstract = metadata['abstractText']
        
        sections = {}
        if metadata["sections"] is not None:
            for sectid in range(len(metadata["sections"])):
                heading = metadata["sections"][sectid]["heading"]
                text = metadata["sections"][sectid]["text"]
                sections[str(heading)] = text
                
        file_path = os.path.join('ReviewAdvisor', 'dataset', folder_name, folder_name+'_paper')
        file_name = paper_id+'_paper.json'
        with open(os.path.join(file_path, file_name), 'r') as f:
            data = json.load(f)
            
        decision = data['decision']
        
        file_path = os.path.join('ReviewAdvisor', 'dataset', folder_name, folder_name+'_review')
        file_name = paper_id+'_review.json'
        with open(os.path.join(file_path, file_name), 'r') as f:
            data = json.load(f)
            
        score = data['reviews']
                
        paper = Paper(paper_id, title, abstract, sections, decision, score)
        return paper
        
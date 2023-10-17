class Paper:
    def __init__(self, ID, TITLE, ABSTRACT, SECTIONS, DECISION, SCORE, REVIEWS=None):
        self.ID = ID
        self.TITLE = TITLE
        self.ABSTRACT = ABSTRACT
        self.SECTIONS = SECTIONS
        self.REVIEWS = REVIEWS
        self.DECISION = DECISION
        self.SCORE = SCORE
        
    def get_paper_content(self):
        content = self.ABSTRACT
        for sect_id in self.SECTIONS:
            if sect_id != 'None':
                content = content + '\n' + self.SECTIONS[sect_id]
        return content
    
    def get_review_content(self):
        content = ''
        for review in self.REVIEWS:
            content = content + '\n' + review['text']
        content = content.strip()
        return content
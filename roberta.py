import PyPDF2
import re
import os
from transformers import RobertaTokenizer, RobertaModel
import faiss
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

class SimilarityChecker():
    def __init__(self, user_file):
        self.user_file = user_file
        if user_file.endswith('.pdf'):
            pdfFileObj = open(user_file, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            x = len(pdfReader.pages)

        if os.path.isfile('processedTextFile.txt'):
            os.remove('processedTextFile.txt')
            
        for y in range(x):
            pageObj = pdfReader.pages[y]
            text = pageObj.extract_text()
            with open('processedTextFile.txt', 'a') as f:
                f.write(text)
        self.user_file = 'processedTextFile.txt'
            

        self.pdfdoc = open(self.user_file, 'r').read().replace('\n', '')
        self.pdfdoc = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', self.pdfdoc)
        self.sequences = list() # list to store multiple lines as sequence
        self.vectors = []
        
        # for i in [4, 3, 2]:
        #     try:
        #         self._init_transformer(i)
        #         break
        #     except RuntimeError:
        #         pass
        self._init_transformer(3)
        averaged_vectors = [torch.mean(vector, dim=0) for vector in self.vectors]

        v_size = [v.size() for v in averaged_vectors][0][0]
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(v_size)) # the size of our vector space
        # index all the documents, we need them as numpy arrays first
        self.index.add_with_ids(
            np.array([t.numpy() for t in averaged_vectors]),
            # the IDs will be 0 to len(documents)
            np.array(range(0, len(self.sequences)))
        )
        
    def _init_transformer(self, n_sentence):
        for i in range(0,len(self.pdfdoc), n_sentence):
            sentences = ' '.join(self.pdfdoc[i:i+n_sentence])
            if len(sentences) > 1200:
                sentences = sentences[:1200]
            self.sequences.append(sentences)
            
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")

        self.vectors = [
            # tokenize the sequence, return it as PyTorch tensors (vectors),
            # and pass it onto the model
            self.model(**self.tokenizer(sequence, return_tensors='pt'))[0].detach().squeeze()
            for sequence in self.sequences
        ]

        
    def encode(self, document: str) -> torch.Tensor:
        tokens = self.tokenizer(document, return_tensors='pt')
        vector = self.model(**tokens)[0].detach().squeeze()
        return torch.mean(vector, dim=0)
    
    def search(self, query: str, k=1):
        encoded_query = self.encode(query).unsqueeze(dim=0).numpy()
        top_k = self.index.search(encoded_query, k)
        scores = top_k[0][0]
        results = [self.sequences[_id] for _id in top_k[1][0]]
        # return list(zip(results, scores))
        if k == 1:
            return scores[0]
    
    

if __name__ == '__main__':
    sc = SimilarityChecker('DEVELOPMENT AND INTEGRATION OF VGG AND DENSE TRANSFER-LEARNING SYSTEMS SUPPORTED WITH DIVERSE LUNG IMAGES FOR DISCOVERY OF THE CORONAVIRUS IDENTITY.pdf')

    with open('userstring.txt', 'r') as f:
        user_text = f.read()
        
    results = sc.search(user_text)
    print(results)

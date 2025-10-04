from dataclasses import dataclass
from typing import List, Tuple
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Doc:
    doc_id: str
    sentences: List[str]

class Retriever:
    def __init__(self, path_jsonl: str):
        self.docs: List[Doc] = []
        with open(path_jsonl, "r") as f:
            for line in f:
                j = json.loads(line)
                self.docs.append(Doc(j["doc_id"], j["sentences"]))
        self.sentences = []
        self.meta = []  # (doc_id, s_idx)
        for d in self.docs:
            for i, s in enumerate(d.sentences):
                self.sentences.append(s)
                self.meta.append((d.doc_id, i))
        self.vect = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        self.X = self.vect.fit_transform(self.sentences)

    def top_k(self, query: str, k: int = 3) -> List[Tuple[str, int, float, str]]:
        q = self.vect.transform([query])
        sims = cosine_similarity(q, self.X)[0]
        idx = sims.argsort()[::-1][:k]
        out = []
        for i in idx:
            doc_id, sidx = self.meta[i]
            out.append((doc_id, sidx, float(sims[i]), self.sentences[i]))
        return out

    def evidence_sentences(self, query: str, k: int = 3) -> List[str]:
        return [s for _,_,_,s in self.top_k(query, k)]


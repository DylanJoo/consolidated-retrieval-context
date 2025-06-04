from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
from nltk import sent_tokenize 
from copy import copy
import random
from utils import normalize_texts, citation_removal

@dataclass
class ReportGenOutput:
    request_id: str
    run_id: str
    collection_ids: List[str]
    raw_report: Optional[str] = None
    cited_report: Optional[str] = None
    references: Dict[str, str] = None
    texts: List[str] = None
    citations: List[str] = None
    """ hard coded the maximum reference numbert = 2 """

    def __post_init__(self):

        if (self.raw_report is None) and (self.cited_report is not None):
            self.cited_report = normalize_texts(self.cited_report)
            self.raw_report = citation_removal(self.cited_report)
            self.set_references(self.references)

            texts = sent_tokenize(self.cited_report)
            self.texts = ["" for _ in range(len(texts))]
            self.citations = [[] for _ in range(len(texts))]

            for idx_text, s in enumerate(texts):
                text, numbers = citation_removal(s, True)
                self.texts[idx_text] = text
                self.set_citations(idx_text, referenceids=numbers)

        if self.texts is None:
            self.texts = sent_tokenize(self.raw_report)

        if self.citations is None:
            self.citations = [[] for _ in range(len(self.texts))]

    def finalize(self):
        sentences = []
        for text, citation in zip(self.texts, self.citations):
            sentences.append({"text": text, "citations": citation[:2]})

        run_id_prefix = self.collection_ids[0].replace('neuclir/1/', '')
        return {
            "request_id": str(self.request_id),
            "run_id": f"{run_id_prefix}_{self.run_id}",
            "collection_ids": self.collection_ids,
            "sentences": sentences
        }

    ## the functions for post-cite -v
    def get_snippets(self, max_word_length=100):
        """ return a list of snippets as query """
        sentences = copy(self.texts)
        snippets = [""]

        while len(snippets[-1].split()) < max_word_length:

            sentence = sentences.pop(0)
            chunk = snippets[-1] + sentence

            if len(chunk.split()) < max_word_length:
                snippets[-1] = chunk
            else:
                snippets.append(sentence)

            if len(sentences) == 0:
                break

        return snippets

    ## the functions for post-cite
    def set_citations(self, idx_text, docids=None, referenceids=None):
        if docids is None:
            docids = []
            for referenceid in referenceids:
                try:
                    docids.append( self.references[referenceid] )
                except:
                    # sometime hallucination with inexisted document
                    print(f"no referenceid {referenceid} from {self.request_id} - {self.collection_ids}")

        self.citations[idx_text] = docids

    # def et_references(self, docids, shuffle=False):
    #     if shuffle:
    #         random.shuffle(docids)
    #     self.references = {str(i+1): docid for i, docid in enumerate(docids)}

    def get_references(self):
        """ return a list of reference ids"""
        assert self.references is not None, 'please add referecnes first'
        reference_mapping = {v: k for k, v in self.references.items()}
        cited_references = []
        for citations in self.citations:
            cited_references.append([reference_mapping[c] for c in citations[:2]])
        return cited_references

    def set_references(self, docids, shuffle=False):
        if shuffle:
            random.shuffle(docids)
        self.references = {str(i+1): docid for i, docid in enumerate(docids)}


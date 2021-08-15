from typing import List, Tuple

import torch
from flair.data import Sentence
from flair.embeddings import Embeddings, DocumentPoolEmbeddings, WordEmbeddings, TransformerDocumentEmbeddings

from nlp_gym.envs.common.observation import BaseObservationFeaturizer
from nlp_gym.envs.quality_estimation.observation import Observation


class SimpleFeaturizer(BaseObservationFeaturizer):

    def __init__(self, doc_embeddings: Embeddings = None, language=None,
                 device: str = "cpu"):
        if language is None:
            language = ["de", "en"]
        self.device = device
        self.language = language
        self._setup_device()
        self.first_embeddings = doc_embeddings if doc_embeddings else DocumentPoolEmbeddings([WordEmbeddings(language[0])])
        self.second_embeddings = doc_embeddings if doc_embeddings else DocumentPoolEmbeddings([WordEmbeddings(language[1])])

    @classmethod
    def from_fasttext(cls) -> 'SimpleFeaturizer':
        return cls(DocumentPoolEmbeddings([WordEmbeddings(cls.language)]))

    # @classmethod
    # def from_transformers(cls) -> 'SimpleFeaturizer':
    #     return cls(TransformerDocumentEmbeddings())

    def init_on_reset(self, question: str, facts: List[str]):
        pass

    def featurize(self, observation: Observation) -> Tuple[torch.Tensor, torch.Tensor]:
        src_embedding = self._get_sentence_embedding(observation.src_sentence, self.language[0])
        tent_embedding = self._get_sentence_embedding(".".join(observation.tent_translated_sentence), self.language[1])
        return src_embedding, tent_embedding

    # def get_observation_dim(self):
    #     embedding = self._get_sentence_embedding("A random sentence to infer dim")
    #     return embedding.shape[1] * 3  # for question, fact and choice

    def _setup_device(self):
        import flair, torch
        flair.device = torch.device(self.device)

    def _get_sentence_embedding(self, text: str, lang: str) -> torch.Tensor:
        text = "..." if len(text) == 0 else text
        sent = Sentence(text)
        doc_embeddings = self.first_embeddings if lang == self.language else self.second_embeddings
        doc_embeddings.embed(sent)
        if len(sent) >= 1:
            embedding = torch.tensor(sent.embedding.cpu().detach().numpy()).reshape(1, -1)
        else:
            embedding = torch.tensor(sent[0].embedding.cpu().detach().numpy()).reshape(1, -1)
        sent.clear_embeddings()
        return embedding
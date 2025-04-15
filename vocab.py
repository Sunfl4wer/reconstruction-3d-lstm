import numpy as np
from collections import Counter
import const
from typing import Optional, Dict, List
from torch.utils.data import Dataset


class Vocab:
    def __init__(self, iterator, min_frequency=2, specials=None):
        self.special_tokens = specials
        self.vocabs = np.asarray([])

        if specials:
            for special_token in self.special_tokens:
                self.vocabs = np.append(self.vocabs, special_token)

        # Count the frequency of each token.
        counter = Counter()
        for token in iterator:
            counter.update(token)

        # Filter out tokens with low frequency
        filtered_counter = {token: freq for token, freq in counter.items() if freq >= min_frequency}

        for token, freq in filtered_counter.items():
            if token not in self.vocabs:
                self.vocabs = np.append(self.vocabs, token)

    def __contains__(self, token: str) -> bool:
        return np.isin(token, self.vocabs)

    def __getitem__(self, token: str) -> int:
        index = np.where(self.vocabs == token)[0]
        if index.size > 0:
          return index[0]
        else:
          return 0

    def __len__(self) -> int:
        return len(self.vocabs)

    def append_token(self, token: str):
        if not self.__contains__(token):
            self.vocabs = np.append(self.vocabs, token)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return np.asarray([self.__getitem__(token) for token in tokens])

    def forward(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def get_default_index(self) -> Optional[str]:
        return self.default_index

    def set_default_index(self, index: int):
        self.default_index = index

    def get_itos(self) -> List[str]:
        return self.vocabs
    
    def get_stoi(self) -> Dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabs)}
        
    def insert_token(self, token: str, index: int):
        if index >= len(self.vocabs):
            raise Exception(f"Index {index} out of range {self.__len__()}")
        if self.__contains__(token):
            raise Exception(f"Token {token} existed!!!")
        self.vocabs = np.insert(self.vocabs, index, token)

    def lookup_token(self, index: int) -> str:
        if index >= len(self.vocabs):
            raise Exception(f"Index {index} out of range {self.__len__()}")
        return self.vocabs[index]
    
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return np.asarray([self.lookup_token(index) for index in indices])

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def build_vocab(train_data):
    image_vocab = Vocab(
        np.asarray([v["image_tokens"] for v in train_data]),
        min_frequency=const.MIN_FREQ,
        specials=const.SPECIAL_TOKENS,
    )

    voxel_vocab = Vocab(
        np.asarray([v["voxel_tokens"] for v in train_data]),
        min_frequency=const.MIN_FREQ,
        specials=const.SPECIAL_TOKENS,
    )
    unk_index = image_vocab[const.UNK_TOKEN]
    image_vocab.set_default_index(unk_index)
    voxel_vocab.set_default_index(unk_index)

    return image_vocab, voxel_vocab
from typing import Iterable, Iterator
import regex as re
import itertools
from .train_bpe import PAT

class Tokenizer:
    def __init__(self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
        ):
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=lambda x:-len(x))
            self.special_token_split_pattern = re.compile('|'.join(re.escape(token) for token in special_tokens))

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        self.bytes_to_id = {v: k for k, v in vocab.items()}
        len_merges = len(merges)
        self.merges_info = {tuple(self.bytes_to_id[x] for x in merge): (i, self.bytes_to_id[b''.join(merge)])  for i, merge in enumerate(merges)}

    @classmethod
    def from_files(cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None,
        ):
        raise NotImplementedError
    
    def _encode_word(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        res = [self.bytes_to_id[bytes([b])] for b in text_bytes]
        while True:
            len_res = len(res)
            candidates = [(*self.merges_info[(res[i], res[i+1])], i) for i in range(len_res-1) if (res[i], res[i+1]) in self.merges_info]
            if not candidates:
                return res
            _, new_id, index_to_merge = min(candidates)
            pair_to_merge = res[index_to_merge], res[index_to_merge+1]
            merged = []
            i = 0
            while i < len_res:
                if i < len_res - 1 and (res[i], res[i+1]) == pair_to_merge:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(res[i])
                    i += 1
            res = merged
                

    def _encode_without_special(self, text: str) -> list[int]:
        words = PAT.findall(text)
        return list(itertools.chain.from_iterable(self._encode_word(word) for word in words))

    def encode(self, text: str) -> list[int]:
        #text = text.replace('\r\n', '\n')
        if not self.special_tokens:
            return self._encode_without_special(text)

        separated = self.special_token_split_pattern.split(text)
        separators = [m.group() for m in self.special_token_split_pattern.finditer(text)]
        assert len(separated) == len(separators) + 1
        separator_ids = [self.bytes_to_id[token.encode("utf-8")] for token in separators]
        separated_ids = [self._encode_without_special(part) for part in separated]
        res = [None] * (len(separated_ids) + len(separator_ids))
        res[::2] = separated_ids
        res[1::2] = [[sid] for sid in separator_ids]
        return list(itertools.chain.from_iterable(res))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[id] for id in ids).decode('utf-8', errors='replace')

if __name__ == "__main__":
    tok = Tokenizer({0 : b's', 1 : b'ss', 2 : b'a', 3 : b'b'}, [], special_tokens=["s", "ss"])
    print(tok.encode("assbssas"))
    print(tok.decode(tok.encode("assbssas")))

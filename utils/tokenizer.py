"""
Implements text tokenization schemes and algorithms. 

Author: Paul Wilson
"""
import os
import torch
import typing as tp
from tqdm import tqdm
from typing import Iterable, Literal
from enum import Enum
import random


__all__ = ["Tokenizer", "BytePairEncodingAlgorithm"]


class Tokenizer:
    """
    Given a vocabulary of tokens, encodes or decodes target strings in and out from
    sequences of integers.
    """
    _TOKEN_KEY = 0 
    UNKNOWN_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    PAD_TOKEN = "<PAD>"
    SPECIAL_TOKENS = [UNKNOWN_TOKEN, START_TOKEN, PAD_TOKEN]

    def __init__(self, vocab: list[str] | dict[str, int]):
        assert len(set(vocab)) == len(
            vocab
        ), "There are duplicate words in the vocabulary!"

        self._token_graph = {}
        self.idx2token = {}

        if isinstance(vocab, list):
            # if we are building from a list, 
            # we get to choose the indexing, and we will
            # put special tokens first
            for token in self.SPECIAL_TOKENS:
                self.add_token(token)

            # then we add the rest of the tokens
            for word in vocab:
                self.add_token(word, exist_ok=True)

        elif isinstance(vocab, dict): 
            # if we are building from a dict,
            # the indexing is already chosen for us
            # and we just need to add the tokens in their
            # respective order
            for word, i in vocab.items():
                self.add_token(word, idx=i)
            
            # then we add the special tokens (if they are not already in the vocab)
            for token in self.SPECIAL_TOKENS:
                self.add_token(token, exist_ok=True)

    @property
    def token2idx(self):
        return {v: k for k, v in self.idx2token.items()}

    @property
    def vocab(self):
        return list(self.token2idx.keys())

    def add_token(self, word, idx=None, exist_ok=False):
        if idx is None: 
            idx = max(self.idx2token.keys(), default=-1) + 1
        if word in self.token2idx:
            if exist_ok:
                return
            else:
                raise ValueError(
                    f"The token {word} is already in the vocabulary"
                    f"at index word {self.token2idx[idx]}"
                )
        d = self._token_graph
        for letter in word:
            d = d.setdefault(letter, {})
        d[self._TOKEN_KEY] = idx
        self.idx2token[idx] = word

    def encode(self, text) -> list[int]:
        # TODO this doesn't work.

        tokens = []
        i = 0
        while i < len(text):
            nodes = [self._token_graph]
            current_word = ""
            while i < len(text) and text[i] in nodes[-1]:
                nodes.append(nodes[-1][text[i]])
                current_word += text[i]
                i += 1
            while len(nodes) > 1 and 0 not in nodes[-1]:
                nodes.pop()
                current_word = current_word[:-1]
                i -= 1
            if current_word == "":
                tokens.append(self.token2idx[self.UNKNOWN_TOKEN])
                i += 1
            else:
                tokens.append(nodes[-1][0])
        return tokens

    def encode_batch(
        self,
        sentences: Iterable[str],
        pad: Literal["right", "left", None] = None,
        add_start_token=False,
        # truncate: Literal["right", "left", None] = None,
        random_offset: bool = False, 
        max_length: int | None = None,
        out_fmt: Literal["list", "numpy", "torch"] = "list"
    ):
        encodings = []
        for sentence in sentences:
            encodings.append(self.encode(sentence))
        
        if add_start_token: 
            encodings_with_start = []
            for encoding in encodings:
                encoding.insert(0, self.token2idx[self.START_TOKEN])
                encodings_with_start.append(encoding)
            encodings = encodings_with_start
        
        longest_sentence = max([len(encoding) for encoding in encodings])        
        pad_token_idx = self.token2idx[self.PAD_TOKEN]
        padded_encodings = []

        for encoding in encodings:
            if pad is not None: 
                num_paddings = longest_sentence - len(encoding)
            else: 
                num_paddings = 0
            if pad == "left" or pad is None:
                encoding = [pad_token_idx] * num_paddings + encoding 
            elif pad == "right": 
                encoding = encoding + [pad_token_idx] * num_paddings
        
            if max_length is not None: 
                if len(encoding) > max_length:
                    # we will have to truncate
                    if random_offset: 
                        # we can randomly shift the sentence as we truncate, 
                        # we can shift it as far as we want to the right as 
                        # long as we don't start including padding tokens. 
                        min_offset = 0
                        max_offset = len(encoding) - max_length - num_paddings
                        if max_offset <= min_offset: 
                            # this means we are already including padding tokens
                            # in the truncated sentence out of necessity
                            offset = 0 
                        else: 
                            # this means that we have space to shift the sentence 
                            # around without including unnecessary padding tokens
                            offset = random.randint(min_offset, max_offset)
                    else: 
                        offset = 0 
                        
                    if pad == "left" or pad is None: 
                        # if there is no padding, the right or left truncation 
                        # is equivalent, so we arbitrarily choose to handle it 
                        # as "right"
                        if offset > 0: 
                            encoding = encoding[-max_length-offset: -offset]
                        else: 
                            encoding = encoding[-max_length:]

                    elif pad == 'right': 
                        encoding = encoding[offset:max_length + offset]

            padded_encodings.append(encoding)
        encodings = padded_encodings

        if out_fmt == 'list': 
            return encodings
        elif out_fmt == 'numpy': 
            import numpy as np 
            return np.array(encodings)
        elif out_fmt == "torch":
            import torch
            return torch.tensor(encodings).long()
        else:
            raise ValueError(f"Invalid out format {out_fmt}")

    def decode(self, encoding):
        return "".join([self.idx2token[i] for i in encoding])

    def __len__(self):
        return len(self.idx2token)

    @classmethod
    def from_json(cls, fpath):
        import json

        with open(fpath, "r") as f:
            vocab = json.load(f)

        return cls(vocab)

    def to_json(self, fpath):
        import json

        with open(fpath, "w") as f:
            json.dump(self.token2idx, f)


class BytePairEncodingAlgorithm:
    tokens: set
    idx2token: list[str]
    token2idx: dict
    tokenfrequencies: dict
    _sorted_tokens_for_encoding: list = None

    UNKNOWN_TOKEN = -1

    def __init__(
        self, allow_whitespace: tp.Literal["prefix_only", True, False], max_tokens=1000
    ):
        """
        Implements Byte Pair Encoding.
        Args:
            allow_whitespace: whether to allow whitespace to be part
                of bigram tokens. If set to True, allows all whitespaces to be fused.
                If false, allows no whitespace tokens to be fused.
                If `prefix only`, allows whitespaces to only be fused as the prefix
                to the bigram (used in GPT2)
            max_tokens (int): The maximum number of tokens this model will attempt to add to the vocabulary
        """
        self.allow_whitespace = allow_whitespace
        self.max_tokens = max_tokens

    def random_fit(self, text_iter, tokenizer=None, max_add=None):
        max_tokens = self.max_tokens
        tokenizer = tokenizer or Tokenizer([]) # start with empty tokenizer

        bar = tqdm(desc="Computing Vocabulary", total=(max_tokens - len(tokenizer)))
        added = 0 
        while len(tokenizer) <= max_tokens and (max_add is None or added <= max_add):
            try: 
                text = next(text_iter) # we have a batch of what could be brand new text
            except StopIteration:
                return tokenizer
            
            for character in set(text): 
                tokenizer.add_token(character, exist_ok=True)
            
            encoding = tokenizer.encode(text)
            bigram_frequencies = self.bigram_frequencies(encoding)
            bigrams = list(bigram_frequencies.keys())
            bigrams.sort(key=lambda bigram: bigram_frequencies[bigram], reverse=True)

            most_frequent_bigram = None
            for bigram in bigrams:
                t1, t2 = bigram
                if self._can_fuse_bigrams(
                    tokenizer.idx2token[t1], tokenizer.idx2token[t2]
                ):
                    most_frequent_bigram = bigram
                    break
            if most_frequent_bigram is None:
                from warnings import warn

                warn(
                    f"BPE fitting finished without reaching max tokens because no remaining "
                    f"bigrams could be fused. "
                )
                return tokenizer
            else:
                i1, i2 = most_frequent_bigram
                replacement_char = tokenizer.idx2token[i1] + tokenizer.idx2token[i2]
                tokenizer.add_token(replacement_char)

            bar.update(1)
            bar.set_postfix_str(f"added `{replacement_char}`")

            added += 1

        return tokenizer

    def fit(self, text, tokenizer=None):
        """
        Uses the byte pair encoding algorithm to find a vocabulary for tokenizing the given
        text.
        """
        self.tokens = set()
        max_tokens = self.max_tokens

        _freqs = {}
        for character in text:
            self.tokens.add(character)
            _freqs.setdefault(character, 0)
            _freqs[character] += 1

        self.idx2token = list(self.tokens)
        self.token2idx = {
            character: idx for idx, character in enumerate(self.idx2token)
        }
        self.tokenfrequencies = {
            self.token2idx[char]: _freqs[char] for char in self.tokens
        }

        tokenization = [self.token2idx[char] for char in text]

        if max_tokens is None:
            return

        from tqdm.auto import tqdm

        bar = tqdm(desc="Computing Vocabulary", total=(max_tokens - len(self.tokens)))

        while len(self.tokens) < max_tokens:
            frequencies = self.bigram_frequencies(tokenization)
            bigrams = list(frequencies.keys())

            # sort list in descending order
            bigrams.sort(key=lambda bigram: frequencies[bigram], reverse=True)

            # find first bigram that does not include whitespace
            most_frequent_bigram = None
            for bigram in bigrams:
                t1, t2 = bigram
                if self._can_fuse_bigrams(self.idx2token[t1], self.idx2token[t2]):
                    most_frequent_bigram = bigram
                    break

            if most_frequent_bigram is None:
                raise ValueError("Fitting failed")

            i1, i2 = most_frequent_bigram
            replacement_char = self.idx2token[i1] + self.idx2token[i2]
            replacement_idx = len(self.tokens)

            self.tokens.add(replacement_char)
            self.idx2token.append(replacement_char)
            self.token2idx[replacement_idx] = replacement_char

            tokenization = self.replace_bigrams(
                tokenization, most_frequent_bigram, replacement_idx
            )

            bar.update(1)
            bar.set_postfix_str(f"added `{replacement_char}`")

        return list(self.tokens)

    def _can_fuse_bigrams(self, prefix, suffix):
        # spaces should come before in tokenization
        if suffix.isspace():
            if self.allow_whitespace == "prefix_only":
                return False
            else:
                return self.allow_whitespace
        # at most one space should be allowed to join
        if prefix.isspace():
            if self.allow_whitespace is True:
                return True
            if suffix[0].isspace():
                return False
            elif prefix != " ":
                return False
            else:
                return False

        return True

    def _sort_key_for_encoding(self, token):
        # primary key should be length, so that long tokens precede short ones.
        # secondary key should be token frequency
        return len(token), self.tokenfrequencies[self.token2idx[token]]

    def replace_bigrams(self, tokenization, bigram: tuple[int, int], replacement: int):
        out = []
        i = 0
        while i < len(tokenization):
            if i + 1 < len(tokenization):
                t1, t2 = tokenization[i : i + 2]
                if (t1, t2) == bigram:
                    out.append(replacement)
                    i += 2
                    self.tokenfrequencies[t1] -= 1
                    self.tokenfrequencies[t2] -= 1
                    self.tokenfrequencies.setdefault(replacement, 0)
                    self.tokenfrequencies[replacement] += 1
                else:
                    out.append(tokenization[i])
                    i += 1
            else:
                out.append(tokenization[i])
                i += 1
        return out

    @staticmethod
    def bigram_frequencies(tokenization):
        frequencies = {}
        for i in range(len(tokenization) - 1):
            bigram = tuple(tokenization[i : i + 2])
            frequencies.setdefault(bigram, 0)
            frequencies[bigram] += 1

        return frequencies



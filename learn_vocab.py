from simple_parsing import parse
import json
from dataclasses import dataclass
import glob
from typing import Literal


@dataclass
class Args:    
    max_tokens: int = 4048
    add_whitespace: Literal[True, False, 'prefix_only'] = 'prefix_only'


def main(args): 
    max_tokens = args.max_tokens

    from bookcorpus import Bookcorpus
    ds = Bookcorpus().as_dataset(split='train')
    def bookcorpus_text_generator(): 
        while True: 
            import random 
            start = random.randint(0, len(ds) - 100)
            text = ds[start:start + 500]['text']
            text = " ".join(text)
            yield text 

    def bible(): 
        with open('data/bible.txt') as f: 
            all_text = f.read()

        len_chunk = 10000    
        while True:
            low = 0 
            high = len(all_text) - len_chunk
            import random 
            startpos = random.randint(low, high)
            yield all_text[startpos:startpos + len_chunk]

    # files = glob.glob('data/*.txt')
    # text = ""
    # for file in files: 
    #     with open(file) as f: 
    #         text = "/n".join([text, f.read()])
    # print(f'Fitting text with {len(text)} characters')

    from torchzero.utils import tokenizer
    algorithm = tokenizer.BytePairEncodingAlgorithm(
        allow_whitespace=args.add_whitespace, max_tokens=args.max_tokens
    )
    tokenizer = tokenizer.Tokenizer([])
    tokenizer = algorithm.random_fit(bible(), tokenizer=tokenizer, max_add=2048)
    tokenizer = algorithm.random_fit(bookcorpus_text_generator(), tokenizer=tokenizer, max_add=None)

    vocab = tokenizer.vocab

    with open(f'vocab_{max_tokens}.json', 'w') as f: 
        json.dump(vocab, f)


if __name__ == '__main__':
    main(parse(Args))
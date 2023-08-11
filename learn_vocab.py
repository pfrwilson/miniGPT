from argparse import ArgumentParser
import json


def main(args): 
    max_tokens = args.max_tokens

    files = ['data/bible.txt', 'data/moby_dick.txt', 'data/sample_openwebtext.txt', 'data/shakespeare.txt']
    text = ""
    for file in files: 
        with open(file) as f: 
            text = "/n".join([text, f.read()])
    print(f'Fitting text with {len(text)} characters')

    from torchzero.utils import tokenizer
    algorithm = tokenizer.BytePairEncodingAlgorithm(allow_whitespace=False, max_tokens=args.max_tokens)
    vocab = algorithm.fit(text)

    with open(f'vocab_{max_tokens}.json', 'w') as f: 
        json.dump(vocab, f)


def parse_args(): 
    parser = ArgumentParser()
    parser.add_argument('--max_tokens', default=512, type=int)
    parser.add_argument('--textfiles', nargs='+')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
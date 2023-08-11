from train import create_model, list_models
import json 
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True, choices=list_models())
    parser.add_argument('--prompt', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--vocab_path', default='vocab_1024.json', type=str)
    parser.add_argument('--rate', default=5, type=float, help='Rate (in Hz) of token printing for the live text generation.')

    return parser.parse_args()


def main(): 
    args = parse_args()

    # Tokenizer
    from torchzero.utils.tokenizer import Tokenizer

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    tokenizer = Tokenizer(vocab)
    tokenizer.add_token("<START>")
    tokenizer.add_token("<PAD>")

    # Model
    model = create_model(args.model_name, tokenizer=tokenizer).to(DEVICE)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    model.eval()
    model.predict_live(prompt=args.prompt, rate=args.rate)


if __name__ == '__main__':
    main()
import json 
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.tokenizer import Tokenizer
from training_logic import registry, TrainingLogic


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True, choices=registry.keys())
    parser.add_argument('--prompt', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--vocab_path', default='vocab_1024.json', type=str)
    parser.add_argument('--rate', default=5, type=float, help='Rate (in Hz) of token printing for the live text generation.')
    parser.add_argument('--device', default='cpu')

    return parser.parse_args()


def main(): 
    args = parse_args()

    # Tokenizer
    with open(args.vocab_path) as f:
        vocab = json.load(f)
    tokenizer = Tokenizer(vocab)
    # tokenizer.add_token("<START>", exist_ok=True)
    # tokenizer.add_token("<PAD>", exist_ok=True)

    # Model
    model, logic = registry[args.model_name](tokenizer=tokenizer)
    logic: TrainingLogic

    model.to(DEVICE)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    model.eval()
    logic.predict_live(model, tokenizer, args.device, prompt=args.prompt, rate=args.rate)


if __name__ == '__main__':
    main()
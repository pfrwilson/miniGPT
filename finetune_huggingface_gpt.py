import argparse
from datetime import date
import datetime
import os

from dataclasses import dataclass, asdict
from utils.experiment_setup import setup
import torch
import sys
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_train_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--textfile",
        type=str,
        default="data/bible.txt",
        help="Path to text file to train on.",
    )
    parser.add_argument(
        "--exp_dir",
        type=lambda x: x.replace('%d', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
        default="logs/debug",
        help="Directory to save logs and checkpoints. (`%%d` is replaced with the current date and time)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the experiment directory if it exists.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    return parser


def get_inference_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model_weights", type=str, required=True, help="Path to model weights."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="1:1 In the beginning, ",
        help="Prompt to start the model with.",
    )
    parser.add_argument("--device", default="cpu", help="Device to run inference on.")
    return parser


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Mode", description="Train or inference. ", dest="mode"
    )
    train_parser = subparsers.add_parser("train", parents=[get_train_parser()])
    inference_parser = subparsers.add_parser(
        "inference", parents=[get_inference_parser()]
    )
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(
            args.device
        )
        model.load_state_dict(torch.load(args.model_weights, map_location="cpu"))
        model.to(args.device)
        run_inference(model, tokenizer, args.device, prompt=args.prompt)


def train(args):
    env = setup(
        args.exp_dir, None, False, False, conf=vars(args), overwrite=args.overwrite
    )

    from dataset import TextFileDataset

    ds = TextFileDataset(args.textfile, chunk_length=500)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(
        args.device
    )
    # for parameter in model.parameters():
    #     parameter.requires_grad_(False)
    # for parameter in model.transformer.wte.parameters():
    #     parameter.requires_grad_(True)
    # for parameter in model.lm_head.parameters():
    #     parameter.requires_grad_(True)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            loader,
            tokenizer,
            os.path.join(env.exp_dir, "sample_outputs.txt"),
            opt,
            args.device,
        )
        env.log_info(f"Finished epoch {epoch} with loss {avg_loss}.")
        torch.save(
            model.state_dict(), os.path.join(env.ckpt_dir, "epoch_{}.pt".format(epoch))
        )


def get_loss(model, tokenizer, batch, device):
    model.train()

    text = batch["text"][0]

    X = tokenizer.encode(text)
    X = torch.tensor(X).unsqueeze(0).to(device)

    X = X.to(device)

    # X is currently shape B * N
    # with entries being the index of the token.

    # To match the input tokens with their target ouputs,
    # we need to shift the targets forward and truncate the outputs like so:
    # tokens:   <PAD> | <PAD> | <START> | h | e | l | l | o |
    # input:    <PAD> | <PAD> | <START> | h | e | l | l | -
    # target:     -       -   |    h    | e | l | l | o |

    targets = X[:, 1:]  # shift inputs backward compared to targets
    X = X[:, :-1]

    # put it through the model
    X = model(X).logits

    # cross entropy loss expects the class scores to be in the second dimension
    X = X.permute(0, 2, 1)
    loss = F.cross_entropy(X, targets)
    return loss


def run_inference(model, tokenizer, device, stream=sys.stdout, limit=None, prompt=None):
    model.eval()

    text = "<|endoftext|>"
    if prompt is not None:
        text += prompt
        stream.write(prompt)
    n = 0
    while True:
        if limit is not None and n > limit:
            break
        n += 1

        inp = tokenizer.encode(text)
        inp = torch.tensor(inp)[None, ...].to(device)
        if len(inp) > 100:
            inp = inp[-100:]
        out = model(inp)
        probs = out.logits.softmax(-1)[0, -1]  # prob distribution over tokens
        sample = torch.multinomial(probs, num_samples=1)
        new_text = tokenizer.decode(sample)
        text += new_text
        stream.write(new_text)
        stream.flush()


def train_one_epoch(model, loader, tokenizer, outputs_file, opt, device):
    model.train()
    avg_loss = 0
    total = 0
    with tqdm(loader) as bar:
        for i, batch in enumerate(bar):
            loss = get_loss(model, tokenizer, batch, device)
            avg_loss += loss.item()
            total += 1

            bar.set_postfix({"loss": loss.item()})
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 1000 == 0:
                with open(outputs_file, "a") as f:
                    f.write("===========================\nITER {}\n\n".format(i))
                    run_inference(
                        model, tokenizer, device, limit=100, prompt=" ", stream=f
                    )
                    f.write("\n===========================")

    return avg_loss / total


if __name__ == "__main__":
    main()

from time import time
import timeit


def my_transformer_vs_huggingface(): 
    import torch 
    from torchinfo import summary
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device {DEVICE}")

    from transformers import BertConfig, BertModel

    model1 = BertModel(BertConfig()).to(DEVICE).encoder 
    print("Instantiated transformers model.")
    summary(
        model1, (1, 256, 768)
    )

    from torchzero.nn import Transformer
    model2 = Transformer(
        n_layers=12, n_heads=12, d_model=768, d_feed_forward=3072,
    ).to(DEVICE)
    print("Instantiated my model")
    summary(
        model2, (1, 256, 768)
    )

    print('=================================')
    print('Testing forward pass.')
    for batch_size in [1, 64, 256]:
        try: 
            print(f'Testing batch_size {batch_size}:  ')
            print('Huggingface model: ')
            sample_input = torch.randn(batch_size, 256, 768, device=DEVICE) 
            print(timeit.timeit("model1(sample_input)", globals=locals(), number=100))
            print('My Model:')
            print(timeit.timeit("model2(sample_input)", globals=locals(), number=100))
        except:
            print(f"Batch size {batch_size} failed! (OOM?)")

my_transformer_vs_huggingface()

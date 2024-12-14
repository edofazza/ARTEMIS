import os

os.environ['HF_HOME'] = './.cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import CLIPTokenizer, CLIPTextModel
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class SummaryDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['video'], row['summary']

    def __len__(self):
        return len(self.df)


def get_embeddings():
    """ DATASET CREATION """
    csv_path = 'animalkingdom_summary.csv'
    dataset = SummaryDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1, persistent_workers=True)
    """ TEXT CLIP ENCODER INITIALIZATION """
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16", device_map='auto')
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    with (torch.no_grad()):
        text_model.eval()
        for videos, summaries in tqdm(dataloader):
            texts = tokenizer(summaries, padding=True, return_tensors="pt", truncation=True
                              ).to('cuda' if torch.cuda.is_available() else 'cpu')
            embeddings = text_model(**texts).pooler_output.detach().cpu().numpy()
            if not os.path.isdir('embeddings'):
                os.mkdir('embeddings')
            for video, embedding in zip(videos, embeddings):
                np.save(os.path.join('embeddings', f'{video}.npy'), embedding)


if __name__ == '__main__':
    get_embeddings()


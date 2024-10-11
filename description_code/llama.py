import os

os.environ['HF_HOME'] = './.cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import gc


class CaptionDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.dataset = self.df['path'].unique()

    def __getitem__(self, idx):
        video = self.dataset[idx]
        return video, self._get_caption(video)

    def __len__(self):
        return len(self.dataset)

    def _get_caption(self, video):
        interested_rows = self.df[self.df['path'] == video]
        text = ''
        for caption in interested_rows.iterrows():
            text += caption[1]['caption']
        return text


def summary_question(captions: str) -> str:
    text = ('Can you write a very short sentence summarizing the information below focusing on the actions of the '
            'animals? Write only the short sentence\n' + captions)
    return text


def verbs_question(captions: str) -> str:
    text = ('List the verbs in the following phrases in the -ing form without repeating, '
            "also add synonyms in the same form. All separated by commas\n") + captions
    return text


def single_run():
    """ DATASET CREATION """
    # csv_path = "/Users/edoardofazzari/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/thesis/captioning/merged_captions.csv"
    csv_path = 'merged_captions.csv'
    dataset = CaptionDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True)
    """ LLAMA INITIALIZATION """
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    token = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto', token=token)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|eot_id|>'),
    ]
    counter = 0
    with torch.no_grad():
        model.eval()
        for videos, captions in tqdm(dataloader):
            correct_list = list()
            for i, caption in enumerate(captions):
                if len(caption) < 10_000:
                    correct_list.append(i)

            if len(correct_list) != 4:
                excluded = [videos[i] for i in range(len(videos)) if i not in correct_list]
                if not os.path.exists('skipped_videos.npy'):
                    np.save('skipped_videos.npy', excluded)
                else:
                    skipped_videos = np.load('skipped_videos.npy')
                    skipped_videos = np.concatenate((skipped_videos, np.array(excluded)))
                    np.save('skipped_videos.npy', skipped_videos)
                videos = [videos[i] for i in correct_list]
                captions = [captions[i] for i in correct_list]

                if len(correct_list) == 0:
                    continue
                # print(len(caption), flush=True, end=',')
            summary_inputs = [
                [{'role': 'user', 'content': summary_question(caption)}] for caption in captions
            ]
            verbs_inputs = [
                [{'role': 'user', 'content': verbs_question(caption)}] for caption in captions
            ]
            all_inputs = summary_inputs + verbs_inputs
            texts = tokenizer.apply_chat_template(all_inputs, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(texts, padding="longest", return_tensors="pt")
            inputs = {key: val.cuda() for key, val in inputs.items()}
            temp_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

            # start_time = time.time()
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]

            df = pd.DataFrame(
                {
                    'video': videos,
                    'summary': gen_text[:len(gen_text) // 2],
                    'caption': gen_text[len(gen_text) // 2:]
                }
            )
            if not os.path.isdir('summarization'):
                os.mkdir('summarization')
            df.to_csv(f'summarization/{counter}.csv', index=False)
            counter += 1

            # Clear CUDA cache
            del inputs
            del gen_tokens
            del gen_text
            del df
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    single_run()

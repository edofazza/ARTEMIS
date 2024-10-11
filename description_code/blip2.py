import os
os.environ['HF_HOME'] = './.cache'

from transformers import BlipForConditionalGeneration, BlipProcessor, Blip2ForConditionalGeneration, Blip2Processor
import torch
from PIL import Image
from torch import nn
import pandas as pd
import tqdm


class Captioning(nn.Module):
    def __init__(self, conditional_text='a photography of'):
        super(Captioning, self).__init__()
        #self.captioning_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
        #self.captioning_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')
        self.captioning_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.captioning_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b').to('cuda')
        self.conditional_text = conditional_text

    def forward(self, images):
        pass

    def predict(self, path: str):
        image = Image.open(path).convert('RGB')
        inputs = self.captioning_processor(image, return_tensors='pt').to('cuda')
        outputs = self.captioning_model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        return self.captioning_processor.decode(outputs[0], skip_special_tokens=True)

    """def predict2(self, dir_path: str):
        images_names = os.listdir(dir_path)
        images = [Image.open(os.path.join(dir_path, image)).convert('RGB') for image in images_names]
        inputs = self.captioning_processor(images, return_tensors='pt').to('cuda')
        outputs = self.captioning_model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        del inputs
        torch.cuda.empty_cache()
        captions_list = []
        for output, name in zip(outputs, images_names):
            captions = self.captioning_processor.decode(output, skip_special_tokens=True)
            captions_list.append((name, captions))

        return captions_list"""

    def predict2(self, dir_path: str):
        images_names = os.listdir(dir_path)

        # Divide the images into groups of maximum 60 elements
        group_size = 20
        num_groups = (len(images_names) + group_size - 1) // group_size

        captions_list = []

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            images_names_group = images_names[start_idx:end_idx]

            images = [Image.open(os.path.join(dir_path, image)).convert('RGB') for image in images_names_group]
            inputs = self.captioning_processor(images, return_tensors='pt').to('cuda')
            outputs = self.captioning_model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)

            for output, name in zip(outputs, images_names_group):
                captions = self.captioning_processor.decode(output, skip_special_tokens=True)
                captions_list.append((name, captions))

            # Free GPU memory after processing the group
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return captions_list


if __name__ == '__main__':
    data = pd.DataFrame(columns=['path', 'frame', 'caption'])
    textbag = Captioning()
    dir_path = 'AnimalKingdom/action_recognition/dataset/image/'
    directories = os.listdir(dir_path)
    #already_saved = pd.read_csv('captions.csv')
    #already_saved = set([file.split('_')[0] for file in already_saved['frame']])
    #directories = list(set(directories) - already_saved)
    for directory in tqdm.tqdm(directories):
        captions_list = textbag.predict2(os.path.join(dir_path, directory))
        torch.cuda.empty_cache()
        for name, caption in captions_list:
            data.loc[len(data)] = [directory, name, caption]
        data.to_csv('captions.csv', index=False)

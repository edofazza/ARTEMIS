import os
import csv
import glob
import numpy as np
from PIL import Image
from torch.utils import data
from itertools import compress


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._data[2]
    

class ActionDataset(data.Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_list = []
        self.random_shift = False

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices
    
    @staticmethod
    def _load_image(directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]
    
    def __getitem__(self, index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        indices = self._sample_indices(record.num_frames)
        return self._get(record, image_names, indices)
    
    def __len__(self):
        return len(self.video_list)
        

class AnimalKingdom(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'action_recognition', 'annotation', mode + '_light.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            if self.anno_path.startswith('train') or self.anno_path.endswith('val_light.csv'):
                reader = csv.DictReader(f, delimiter=';')
            else:
                reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                ovid = row['video_id']
                labels = row['labels']
                path = os.path.join(self.path, 'action_recognition', 'dataset', 'image', ovid)
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoRecord([path, count, labels])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        summaries = np.load(os.path.join('embeddings', image_names[0].split('.')[0].split('_')[0] + '.npy'))
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = np.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        return process_data, np.array(summaries), label
        

class Charades(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'Charades', 'Charades_v1_' + mode + '.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    @staticmethod
    def _cls2int(x):
        return int(x[1:])

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                actions = row['actions']
                if actions == '': continue
                vid = row['id']
                path = os.path.join(self.path, 'Charades_v1_rgb', vid)
                files = sorted(os.listdir(path))
                num_frames = len(files)
                fps = num_frames / float(row['length'])
                labels = np.zeros((num_frames, self.num_classes), dtype=bool)
                actions = [[self._cls2int(c), float(s), float(e)] for c, s, e in [a.split(' ') for a in actions.split(';')]]
                for frame in range(num_frames):
                    for ann in actions:
                        if frame/fps > ann[1] and frame/fps < ann[2]: labels[frame, ann[0]] = 1
                idx = labels.any(1)
                num_frames = idx.sum()
                file_list += [list(compress(files, idx.tolist()))]
                video_list += [VideoRecord([path, num_frames, labels[idx]])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = record.label[indices].any(0).astype(np.float32)
        return process_data, label


class MammalNet(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode

        self.anno_path = os.path.join(self.path, f'annotation/{mode}ing.csv')

        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                ovid = row['video_id']
                labels = row['behavior']
                path = os.path.join(self.path, 'dataset', 'image', ovid)
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = int(labels)
                video_list += [VideoRecord([path, count, labels])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        summaries = np.load(os.path.join('mammalnet_embeddings', record.path.split('/')[-1] + '.npy'))
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
                #print('CORRECT: ', record.path, image_names[idx])
            except:
                print(record.path, image_names, idx, flush=True)
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        #label = np.zeros(self.num_classes)  # need to fix this hard number
        #label[record.label] = 1.0
        return process_data, np.array(summaries), record.label

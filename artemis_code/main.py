import os
import torch
import string
import random
import numpy as np
from utils.utils import read_config

def main(args):
    os.environ['HF_HOME'] = './.cache'
    if(args.seed>=0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("[INFO] Setting SEED: " + str(args.seed))   
    else:
        print("[INFO] Setting SEED: None")

    if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type:", str(device), flush=True)

    config = dict()
    config['path_dataset'] = '.'  # MODIFIED, substituted get_config
    if args.dataset == "animalkingdom":
        dataset = 'AnimalKingdom'
    elif args.dataset == "baboonland":
        dataset = 'baboonland'
    elif args.dataset == "mammalnet":
        dataset = 'mammalnet'
    elif args.dataset == "ava":
        dataset = 'AVA'
    else:
        dataset = string.capwords(args.dataset)
    path_data = os.path.join(config['path_dataset'], dataset)
    print("[INFO] Dataset path:", path_data, flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    train_loader = manager.get_train_loader(train_transform)
    print("[INFO] Train size:", str(len(train_loader.dataset)), flush=True)

    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform)
    print("[INFO] Test size:", str(len(val_loader.dataset)), flush=True)

    # criterion or loss
    import torch.nn as nn
    if args.dataset in ['animalkingdom', 'baboonland']:
        criterion = nn.BCEWithLogitsLoss()
    elif args.dataset in ['mammalnet']:
        criterion = nn.CrossEntropyLoss()

    # evaluation metric
    if args.dataset in ['animalkingdom', 'baboonland']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision'
    elif args.dataset in ['mammalnet']:
        from torchmetrics.classification import MulticlassAccuracy
        eval_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
        eval_metric_string = 'Multiclass Accuracy'

    name = 'variation1_'
    if args.recurrent != 'none':
        name += args.recurrent + '_'
    if args.residual:
        name += 'residual_'
    if args.sumresidual:
        name += 'sumresidual_'
    if args.backboneresidual:
        name += 'backboneresidual_'
    if args.linear2residual:
        name += 'linear2residual_'
    if args.imageresidual:
        name += 'imageresidual_'
    if args.relu:
        name += 'relu'
    print('MODEL NAME: ', name, flush=True)
    # model
    if args.contrastive == 'none':
        model_args = (train_loader, val_loader, criterion, eval_metric, class_list, args.test_every, args.distributed, device,
                      args.recurrent if args.recurrent != 'none' else None,
                      args.fusion,
                      args.residual,
                      args.relu,
                      args.sumresidual,
                      args.backboneresidual,
                      args.linear2residual,
                      args.imageresidual)
    else:
        model_args = (
        train_loader, val_loader, criterion, eval_metric, class_list, args.test_every, args.distributed, device,
        args.recurrent if args.recurrent != 'none' else None,
        args.fusion,
        args.residual,
        args.relu,
        args.sumresidual,
        args.backboneresidual,
        args.linear2residual,
        args.imageresidual,
        args.contrastive)
    if args.model == 'artemis':
        from models.artemis import ARTEMISExecutor
        executor = ARTEMISExecutor(*model_args)
    elif args.model == 'artemis_contrastive':
        from models.artemis_contrastive import ARTEMISContrastiveExecutor
        executor = ARTEMISContrastiveExecutor(*model_args)

    executor.train(args.epoch_start, args.epochs)
    eval = executor.test()

    if args.contrastive == 'none':
        executor.save(f'models/{name}')
    else:
        executor.save(f'contrastive/{args.contrastive}_{name}')

    #executor.save('variation1_bilstm_epoch200_adamw')
    print("[INFO] " + eval_metric_string + ": {:.2f}".format(eval * 100), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="animalkingdom", help="animalkingdom")
    parser.add_argument("--model", default="artemis", help="Model: artemis, artemis_contrastive")
    parser.add_argument("--total_length", default=10, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the mini-batch")
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of torchvision workers used to load data (default: 8)")
    parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--test_part", default=6, type=int, help="Test partition for Hockey dataset")
    parser.add_argument("--zero_shot", default=False, type=bool, help="Zero-shot or Fully supervised")
    parser.add_argument("--split", default=1, type=int, help="Split 1: 50:50, Split 2: 75:25")
    parser.add_argument("--train", default=False, type=bool, help="train or test")

    parser.add_argument("--recurrent", default='none', type=str, help="bilstm / gru / conv")
    parser.add_argument("--fusion", default='normal', type=str, help="TODO")
    parser.add_argument("--residual", action='store_true', help="PosEncoder residual")
    parser.add_argument("--relu", action='store_true', help="ReLU layers")
    parser.add_argument("--sumresidual", action='store_true', help="Summary residual")
    parser.add_argument("--backboneresidual", action='store_true', help="Backbone residual")
    parser.add_argument("--linear2residual", action='store_true', help="Linear2 residual")
    parser.add_argument("--imageresidual", action='store_true', help="Image residual")

    parser.add_argument("--contrastive", default='none', type=str, help="none / contrastive / cosine / cca")
    args = parser.parse_args()

    main(args)

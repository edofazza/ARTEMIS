import os
os.environ['MPLCONFIGDIR'] = '/workdir/.matplotlib_local'
os.environ['HF_HOME'] = '/workdir/.cache'

import string
import torch
import numpy as np
import random
import time
from ensemble.ensemble import Ensemble
from ensemble.genetic import GeneticEnsemble
from models.artemis import ARTEMIS
from models.artemis_contrastive import ARTEMISContrastive


def main(args):
    if (args.seed >= 0):
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

    if (torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type:", str(device), flush=True)

    config = dict()
    config['path_dataset'] = '.'  # MODIFIED, substituted get_config

    config = dict()
    config['path_dataset'] = '.'  # MODIFIED, substituted get_config
    if args.dataset == "animalkingdom":
        dataset = 'AnimalKingdom'
    elif args.dataset == "baboonland":
        dataset = 'baboonland'
    elif args.dataset == "mammalnet":
        dataset = 'mammalnet'
    else:
        dataset = string.capwords(args.dataset)
    path_data = os.path.join(config['path_dataset'], dataset)
    print("[INFO] Dataset path:", path_data, flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    if args.train:
        # NO TRANSFORMATION
        train_transform = manager.get_test_transforms()
        data_loader = manager.get_cross_loader(train_transform, args.k)
        print(f"[INFO] Cross val {args.k} size:", str(len(data_loader.dataset)), flush=True)
    else:
        # test data
        val_transform = manager.get_test_transforms()
        data_loader = manager.get_test_loader(val_transform)
        print("[INFO] Test size:", str(len(data_loader.dataset)), flush=True)

    # MODEL DEFINITIONS
    model1 = ARTEMIS(class_embed=torch.rand((140, 512)),  # BEST 1 200 epoches
                     num_frames=16,
                     recurrent='conv',
                     fusion='normal',
                     residual=True,
                     relu=False,
                     summary_residual=False,
                     backbone_residual=True,
                     linear2_residual=False,
                     image_residual=True).to(device)
    model1.load_state_dict(torch.load('models/variation1_conv_residual_backboneresidual_imageresidual_.pth'))

    model2 = ARTEMISContrastive(class_embed=torch.rand((140, 512)),  # BEST 3 cosine
                                num_frames=16,
                                recurrent='conv',
                                fusion='normal',
                                residual=True,
                                relu=False,
                                summary_residual=False,
                                backbone_residual=True,
                                linear2_residual=True,
                                image_residual=False).to(device)
    model2.load_state_dict(torch.load('models/variation1_conv_residual_backboneresidual_linear2residual_.pth'))

    model3 = ARTEMIS(class_embed=torch.rand((140, 512)),  # ARTEMIS cosine
                     num_frames=16,
                     recurrent='bilstm',
                     fusion='normal',
                     residual=True,
                     relu=False,
                     summary_residual=True,
                     backbone_residual=True,
                     linear2_residual=True,
                     image_residual=True).to(device)
    model3.load_state_dict(
        torch.load('models/variation1_bilstm_residual_sumresidual_backboneresidual_linear2residual_imageresidual_.pth'))

    models = [model1, model2, model3]

    ens = Ensemble(models=models, data_loader=data_loader, device=device, num_labels=num_classes)
    if args.type == 'ensemble':
        initial_time = time.time()
        #eval = ens.test(weights=[0.35527880632037867, 0.33102491647831506, 0.31369627720130633])    # FOLD 0
        #eval = ens.test(weights=[0.34381520529054543, 0.35082624709176513, 0.3053585476176895])  # FOLD 1
        #eval = ens.test(weights=[0.34348380125277234, 0.3341105545108527, 0.32240564423637497])  # FOLD 2
        eval = ens.test(weights=[0.3578997689238905, 0.3847545988164762, 0.25734563225963336])  # FOLD 3
        #eval = ens.test(weights=[0.3468346961521773, 0.3666702635596512, 0.2864950402881715])  # FOLD 4
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)
    elif args.type == 'ga':
        initial_time = time.time()
        genetic_ensemble = GeneticEnsemble(ens, 10, 0.5, 0.2, 2, fold=args.k)
        genetic_ensemble.train()
        final_time = time.time()
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)
    elif args.type == 'rl_static':
        from ensemble.reinforcement_static import train
        train(ens)
    elif args.type == 'rl_dynamic':
        from ensemble.reinforcement_dynamic import train, test
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(args.seed)
        if args.train:
            train(models, data_loader, device, args.k)
        else:
            test(models, data_loader, device)
    else:
        print("[ERROR] Unknown type")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training script for ensemble")
    parser.add_argument('--type', type=str, default='ensemble', help="ensemble/ga/rl")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--dataset", default='animalkingdom', type=str, help='animalkingdom')
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=16, type=int, help="Size of the mini-batch")
    parser.add_argument("--num_workers", default=2, type=int,
                        help="Number of torchvision workers used to load data (default: 2)")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--train", action='store_true', help="train/test")
    parser.add_argument("--k", default=0, type=int, help="set between 0 and 4 (included)")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    args = parser.parse_args()

    main(args)

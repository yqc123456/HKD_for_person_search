import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data
from eval_reid.eval_reid import testwithVer2, testwithVer2_CUHKSYSU, testwithVer2_PRW
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.seq_reid import Seq_Reid
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from datasets.market1501 import Loaders


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = Seq_Reid(cfg)
    model.to(device)

    print("Loading data")
    loaders = Loaders(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)

        gallery_feats, query_feats = evaluate_performance(
            model,
            loaders.duke_gallery_loader,
            loaders.duke_query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        if args.dataset == 'cuhksysu':
            testwithVer2_CUHKSYSU(cfg, loaders, gallery_feats, query_feats)
        elif args.dataset == 'G2APS':
            testwithVer2(cfg, loaders, gallery_feats, query_feats)
        elif args.dataset == 'PRW':
            testwithVer2_PRW(cfg, loaders, gallery_feats, query_feats)
        else:
            print('wrong dataset')
            exit(-4)

        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, model, optimizer, loaders, device, epoch, tfboard)
        lr_scheduler.step()

        if epoch % cfg.eval_interval == 0 and epoch > 0:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

            gallery_feats, query_feats = evaluate_performance(
                model,
                loaders.duke_gallery_loader,
                loaders.duke_query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )
            if args.dataset == 'cuhksysu':
                testwithVer2_CUHKSYSU(cfg, loaders, gallery_feats, query_feats)
            elif args.dataset == 'G2APS':
                testwithVer2(cfg, loaders, gallery_feats, query_feats)
            elif args.dataset == 'PRW':
                testwithVer2_PRW(cfg, loaders, gallery_feats, query_feats)
            else:
                print('wrong dataset')
                exit(-4)

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    # dataset configuration
    parser.add_argument('--duke_path', type=str, default='path/to/occluded/duke')
    parser.add_argument('--dataset', type=str, default="G2APS")
    parser.add_argument('--train_dataset', type=str, default='duke', help='occluded_duke')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--p', type=int, default=16, help='person count in a batch')
    parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)

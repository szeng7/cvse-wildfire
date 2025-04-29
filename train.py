import argparse
import os
import datetime

from functions import get_dataset, train, Loss
from models import MLP_CNN, CAE, NDWS_CAE, UNet, UNet_Light

TRAIN_PATTERN="data_full/train*"
EVAL_PATTERN="data_full/eval__000*"
TEST_PATTERN="data_full/test*"

NUM_FEATURES=16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLPCNN', choices=['MLPCNN', 'CAE', 'NDWS_CAE','UNET','UNET_L'], help='model/architecture to run training with')
    parser.add_argument('--data-dir', type=str, default='./data', help='directory that contains the data')
    parser.add_argument('--num-steps', type=int, default=100, help='number of steps to run for training')
    parser.add_argument('--loss', type=str, default='BCE', choices=['BCE', 'weighted_BCE', 'focal'], help='loss function to use during training')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data batches, use --shuffle to enable shuffling, omit to disable')
    parser.add_argument('--seed', type=int, default=19, help='random seed for shuffling')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='directory for model checkpoints')
    args = parser.parse_args()

    train_file_pattern = os.path.join(args.data_dir, TRAIN_PATTERN)

    train_dataset = get_dataset(
        train_file_pattern,
        data_size=64, #full tile size, 64kmx64km
        sample_size=32, #random 32x32 crops for training
        batch_size=args.batch_size,
        num_in_channels=NUM_FEATURES,
        compression_type=None,
        clip_and_normalize=False,
        clip_and_rescale=False,
        random_crop=True, #randomly cropping subregions helps with reducing overfitting/better generalization
        center_crop=False
        )

    eval_file_pattern = os.path.join(args.data_dir, EVAL_PATTERN)

    eval_dataset = get_dataset(
        file_pattern=eval_file_pattern,
        data_size=64,
        sample_size=64, #use full tile for evaluation now
        batch_size=args.batch_size,
        num_in_channels=NUM_FEATURES,
        compression_type=None,
        clip_and_normalize=False,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=True #don't think this matters since no cropping (sample size 64) but in case it does, for consistency
    )

    if args.model == "MLPCNN":
        model = MLP_CNN(input_shape=(None, None, 16))
    elif args.model == "CAE":
        model = CAE(input_shape=(None, None, 16))
    elif args.model == "NDWS_CAE":
        model = NDWS_CAE(input_shape=(None, None, 16))
    elif args.model == "UNET":
        model = UNet(input_shape=(None, None, 16))
    elif args.model == "UNET_L":
        model = UNet_Light(input_shape=(None, None, 16))
    else:
        raise ValueError(f"Model provided not supported yet: {args.model}")
    
    if args.loss == "BCE":
        loss_type=Loss.BCE
    elif args.loss == "weighted_BCE":
        loss_type=Loss.WEIGHTED_BCE
    elif args.loss == "focal":
        loss_type=Loss.FOCAL
    else:
        raise ValueError(f"Provided loss not supported: {args.loss}")
    
    train(model, train_dataset, eval_dataset, checkpoint_dir=args.checkpoint_dir, loss_type=loss_type, label=f"{args.model}-{args.loss}", num_steps=args.num_steps)


if __name__ == '__main__':
    main()
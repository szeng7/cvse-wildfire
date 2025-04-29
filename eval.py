import argparse
import os
import datetime

from functions import get_dataset, Loss, evaluate_model
from models import MLP_CNN, CAE, NDWS_CAE, UNet, UNet_Light
import tensorflow as tf

TRAIN_PATTERN="data_full_train*"
EVAL_PATTERN="data_full_eval*"
TEST_PATTERN="data_full_test*"

NUM_FEATURES=16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLPCNN', choices=['MLPCNN', 'CAE', 'NDWS_CAE','UNET','UNET_L'], help='model/architecture to run training with')
    parser.add_argument('--data-dir', type=str, default='./data', help='directory that contains the data')
    parser.add_argument('--checkpoint-dir', type=str, help='checkpoint containing pretrained model weights')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    args = parser.parse_args()

    test_file_pattern = os.path.join(args.data_dir, TEST_PATTERN)

    test_dataset = get_dataset(
        file_pattern=test_file_pattern,
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
        model = UNet(input_shape=(None, None, NUM_FEATURES))
    elif args.model == "UNET_L":
        model = UNet_Light(input_shape=(None, None, NUM_FEATURES))
    else:
        raise ValueError(f"Model provided not supported yet: {args.model}")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.restore(args.checkpoint_dir).expect_partial()
    
    evaluate_model(model, test_dataset)
    
if __name__ == '__main__':
    main()
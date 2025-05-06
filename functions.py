import re
from typing import Dict, List, Optional, Text, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf
from tensorflow.keras import layers, Model

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, jaccard_score, mean_absolute_error
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from enum import Enum
import os

# --------------------------
# Training
# --------------------------



class Loss(Enum):
    BCE = "bce"
    WEIGHTED_BCE = "weighted_bce"
    FOCAL = "focal"
    DICE = "dice"
    TVERSKY = "tversky"

def dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Dice Loss = 1 − (2 * intersection + ε) / (sum(y_true) + sum(y_pred) + ε)
    Good for highly imbalanced foreground (fire) vs background.
    """
    # flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice_coeff = (2.0 * intersection + epsilon) / (denom + epsilon)
    return 1.0 - dice_coeff

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, epsilon=1e-6):
    """
    Tversky Loss = 1 − TI, where
      TI = intersection / (intersection + α * FP + β * FN)
    α<β emphasizes penalizing false negatives harder.
    """
    # flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # true positives, false negatives, false positives
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)

    tversky_index = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    return 1.0 - tversky_index


def weighted_bce(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = tf.where(tf.equal(y_true, 1.0), 5.0, 1.0)  #weigh positive class more (it has less appearances and is fire positive)

    #make shapes match
    if len(bce.shape) == 3:
        bce = tf.expand_dims(bce, axis=-1)
    if len(weights.shape) == 3:
        weights = tf.expand_dims(weights, axis=-1)

    return tf.reduce_mean(bce * weights)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    #bound predictions between 0 and 1
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    modulating_factor = tf.pow((1 - p_t), gamma)

    loss = alpha_factor * modulating_factor * bce
    return tf.reduce_mean(loss)

def evaluate_model(model, eval_dataset, seed=19):
    y_probs, y_preds, y_trues = [], [], []
    for x_batch, y_batch in eval_dataset:
        probs = model(x_batch, training=False)
        preds = tf.cast(probs > 0.5, tf.float32)
        y_probs.append(probs)
        y_preds.append(preds)
        y_trues.append(y_batch)

    y_probs = tf.concat(y_probs, axis=0)
    y_preds = tf.concat(y_preds, axis=0)
    y_trues = tf.concat(y_trues, axis=0)

    # classic metrics
    auc_pr     = calc_auc_pr(y_probs, y_trues)
    fnr        = calc_false_negative_rate(y_preds, y_trues)
    jaccard    = calc_jaccard(y_preds, y_trues)

    # sample a few tiles for shape errors
    np.random.seed(seed)
    idxs = np.random.choice(len(y_preds), size=min(5, len(y_preds)), replace=False)

    mae_vals     = []
    chamfer_vals = []
    for i in idxs:
        mae = calc_mae_fire_front(y_preds[i], y_trues[i])
        cd  = calc_chamfer_distance(y_preds[i], y_trues[i])
        if not np.isnan(mae):     mae_vals.append(mae)
        if not np.isnan(cd):      chamfer_vals.append(cd)

    fire_mae     = np.mean(mae_vals) if mae_vals else float("nan")
    chamfer_dist = np.mean(chamfer_vals) if chamfer_vals else float("nan")

    print(f"[Eval] AUC-PR: {auc_pr:.4f}, FNR: {fnr:.4f}, IoU: {jaccard:.4f}, "
          f"FireFront MAE: {fire_mae:.2f}, Chamfer: {chamfer_dist:.2f}")

def save_checkpoint(model, optimizer, checkpoint_dir, step, label=None):

    if label:
        subfolder = os.path.join(checkpoint_dir, label, f"step_{step:04d}")
    else:
        subfolder = os.path.join(checkpoint_dir, f"step_{step:04d}")
    os.makedirs(subfolder, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_path = os.path.join(subfolder, "ckpt")
    ckpt.write(ckpt_path)

    print(f"Checkpoint saved at: {ckpt_path}")
    return ckpt_path

def train(model, train_dataset, eval_dataset, checkpoint_dir, loss_type, label=None, num_steps=100, optimizer=None, eval_interval=50):
    
    if loss_type == Loss.BCE:
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    elif loss_type == Loss.WEIGHTED_BCE:
        loss_fn = weighted_bce
    elif loss_type == Loss.FOCAL:
        loss_fn = focal_loss
    elif loss_type == Loss.DICE:
        loss_fn = dice_loss
    elif loss_type == Loss.TVERSKY:
        loss_fn = tversky_loss
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")

    acc_metric = tf.keras.metrics.BinaryAccuracy()

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,clipnorm=1.0)

    print(f"Running pre-training evaluation...")
    evaluate_model(model, eval_dataset)
    save_checkpoint(model, optimizer, checkpoint_dir=checkpoint_dir, step=0, label=label)
    
    train_dataset = train_dataset.repeat()
    print("ready to start trainig...")
    for step, (x_batch, y_batch) in enumerate(train_dataset.take(num_steps)):
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss = loss_fn(y_batch, preds)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        acc_metric.update_state(y_batch, preds)

        if step % eval_interval == 0:
            acc = acc_metric.result().numpy()
            print(f"Step {step:03d}: Loss = {loss:.4f} | Accuracy = {acc:.4f}")
            acc_metric.reset_state()
            
            evaluate_model(model, eval_dataset)
            
            save_checkpoint(model, optimizer, checkpoint_dir=checkpoint_dir, step=step, label=label)
            
    print(f"Running post-training evaluation...")
    evaluate_model(model, eval_dataset)
    save_checkpoint(model, optimizer, checkpoint_dir=checkpoint_dir, step=num_steps, label=label)

            
            

# --------------------------
# Evaluation
# --------------------------

def calc_auc_pr(y_pred_proba, y_true):
    """
    Args:
        y_pred_proba: predicted probabilities, shape (32,32,1)
        y_true: ground truth, shape (32,32,1)
    Returns:
        Single value from 0 to 1. Higher value (closer to 1) indicates the model successfully identifies fires
        while minimizing unnecessary false alarms 
    """
    y_pred_flat = y_pred_proba.numpy().flatten()
    y_true_flat = y_true.numpy().flatten()
    
    #mask out uncertain labels (-1)
    valid_mask = y_true_flat != -1
    y_pred_flat = y_pred_flat[valid_mask]
    y_true_flat = y_true_flat[valid_mask]
    
    #if no positive or negative labels left
    if len(np.unique(y_true_flat)) < 2:
        return 0.0
    
    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat) #not using third output of thresholds
    return auc(recall, precision)

def calc_false_negative_rate(y_pred, y_true):
    """
    Args:
        y_pred: binary predictions, shape (32,32,1)
        y_true: ground truth, shape (32,32,1)
    Returns:
        Single value from 0 to 1. Lower value (closer to 0) indicates that the model predicts less false negatives. In this context,
        false negatives (inaccurately predicting no wildfire) is catastrophic and avoiding them should be a priority.
    """
    y_pred_flat = y_pred.numpy().flatten()
    y_true_flat = y_true.numpy().flatten()
    
    #mask out uncertain labels (-1)
    valid_mask = y_true_flat != -1
    y_pred_flat = y_pred_flat[valid_mask]
    y_true_flat = y_true_flat[valid_mask]
    
    _, _, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel() #not using first two outputs of true negative and false positives
    return fn / (fn + tp + 1e-8) #avoid division by zero

def calc_jaccard(y_pred, y_true):
    """
    Args:
        y_pred: binary predictions, shape (32,32,1)
        y_true: ground truth, shape (32,32,1)
    Returns:
        Single value from 0 to 1. High value (closer to 1) indicates that the model better captures the overall extent
        and shape of the entire fire (value of 0.8 indicates predicted overlaps with 80% of actual)
    """
    y_pred_flat = y_pred.numpy().flatten()
    y_true_flat = y_true.numpy().flatten()
    
    #mask out uncertain labels (-1)
    valid_mask = y_true_flat != -1
    y_pred_flat = y_pred_flat[valid_mask]
    y_true_flat = y_true_flat[valid_mask]
    
    return jaccard_score(y_true_flat, y_pred_flat)

def calc_mae_fire_front(y_pred, y_true, unmatched_penalty=50):
    """
    NOTE: can rewrite this if found to be too aggressive while evaluating models (since this does one-to-one, which penalizes the model
    heavily when it predicts a fire front pixel that doesn't actually exist in the true mask)
    Args:
        y_pred: binary predictions, shape (32,32,1)
        y_true: ground truth, shape (32,32,1)
        unmatched_penalty: penalty for when a predicted fire front pixel has no corresponding match in the true mask, lower if too aggressive
    Returns:
        Single value from 0 to inf (technically in this case, it's limited by the input grid size). A lower value (closer to 0) indicates
        that the predicted fire front is closer (avg number of pixels) to the actual fire front. This is important because understanding the
        fire front/leading edge of the fire is critical for firefighting, evacuation and resource allocation decisions. 
    """
    def extract_fire_front(mask):
        """
        Args:
            mask: binary 2D nparray
        Returns:
            coordinates of fire front pixels
        """
        mask = mask.numpy().squeeze().astype(np.uint8) #ensure 2D
        #ignore uncertain labels
        mask[mask == -1] = 0
        edges = cv2.Canny(mask * 255, 100, 200) #canny gets firefront using edge detection
        front_coords = np.column_stack(np.where(edges > 0)) #returns (rows, columns), ie (y,x)
        return front_coords
    
    pred_front = extract_fire_front(y_pred)
    true_front = extract_fire_front(y_true)

    if len(pred_front) == 0 or len(true_front) == 0:
        return np.nan #no front to calculate using
    
    #pad smaller away with dummy points far away to enforce one-to-one matching, penalizing inaccurate edges
    max_len = max(len(pred_front), len(true_front))
    #far point to penalize unmatched assignments
    dummy_point = np.array([[unmatched_penalty, unmatched_penalty]]) 

    if len(pred_front) < max_len:
        pad = np.tile(dummy_point, (max_len - len(pred_front), 1))
        pred_front = np.vstack([pred_front, pad])
    elif len(true_front) < max_len:
        pad = np.tile(dummy_point, (max_len - len(true_front), 1))
        true_front = np.vstack([true_front, pad])

    #calc pairwise distances
    cost_matrix = cdist(pred_front, true_front)

    #solve optimal one to one matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_dists = cost_matrix[row_ind, col_ind]

    return np.mean(matched_dists)

def calc_chamfer_front(y_pred, y_true, unmatched_penalty=50):
    """
    Chamfer distance calculated *only* on the fire front (edges) of the binary masks.
    Uses Canny edge detection to extract front pixels, then averages NN distances both ways.
    """
    def extract_front_coords(mask):
        # Turn to 2D uint8, zero out uncertain labels, run Canny to find edges
        arr = mask.numpy().squeeze().astype(np.uint8)
        arr[arr == -1] = 0
        edges = cv2.Canny(arr * 255, 100, 200)
        return np.column_stack(np.where(edges > 0))

    # Get front pixel coords for both prediction and truth
    pred_front = extract_front_coords(y_pred)
    true_front = extract_front_coords(y_true)

    if len(pred_front) == 0 or len(true_front) == 0:
        return np.nan

    # Pad the smaller set so we enforce 1:1 matching
    max_len = max(len(pred_front), len(true_front))
    dummy = np.array([[unmatched_penalty, unmatched_penalty]])
    if len(pred_front) < max_len:
        pred_front = np.vstack([pred_front,
                                np.tile(dummy, (max_len - len(pred_front), 1))])
    if len(true_front) < max_len:
        true_front = np.vstack([true_front,
                                np.tile(dummy, (max_len - len(true_front), 1))])

    # Build KD-trees and compute bidirectional nearest‐neighbor distances
    tree_true = cKDTree(true_front)
    tree_pred = cKDTree(pred_front)
    d_pt, _ = tree_true.query(pred_front)
    d_tp, _ = tree_pred.query(true_front)

    # Return the mean of both directions
    return float((d_pt.mean() + d_tp.mean()) / 2.0)

# --------------------------
# Constants
# --------------------------

INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask',
                  'u_component_of_wind_10m_above_ground',
                  'v_component_of_wind_10m_above_ground',
                  'temperature_2m_above_ground',
                  'precipitable_water_entire_atmosphere']

OUTPUT_FEATURES = ['FireMask']

DATA_STATS = {
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    'pdsi': (-6.1298, 7.8760, -0.0053, 2.6823),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    'pr': (0.0, 44.5304, 1.7398051, 4.4828),
    'sph': (0.0, 1.0, 0.0071658953, 0.0042835088),
    'th': (0.0, 360.0, 190.3298, 72.5985),
    'tmmn': (253.15, 298.9489, 281.08768, 8.9824),
    'tmmx': (253.15, 315.0923, 295.17383, 9.8155),
    'vs': (0.0, 10.0243, 3.8501, 1.4110),
    'erc': (0.0, 106.2489, 37.3263, 20.8460),
    'population': (0.0, 2534.0630, 25.5314, 154.7233),
    'PrevFireMask': (-1.0, 1.0, 0.0, 1.0),
    'u_component_of_wind_10m_above_ground': (-0.1272, 1.4860, 0.6746, 0.3910),
    'v_component_of_wind_10m_above_ground': (-0.6535, 0.9800, 0.1719, 0.3948),
    'temperature_2m_above_ground': (15.2312, 18.6066, 16.9189, 0.8326),
    'precipitable_water_entire_atmosphere': (16.0138, 18.7311, 17.3379, 0.6677),
    'FireMask': (-1.0, 1.0, 0.0, 1.0),
}

# --------------------------
# Fire Label Mapping
# --------------------------

def map_fire_labels(labels):
    non_fire = tf.where(
        tf.logical_or(tf.equal(labels, 3), tf.equal(labels, 5)),
        tf.zeros_like(labels), -1 * tf.ones_like(labels))
    fire = tf.where(tf.greater_equal(labels, 7), tf.ones_like(labels), non_fire)
    return tf.cast(fire, dtype=tf.float32)

# --------------------------
# Preprocessing Utilities
# --------------------------

def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(
        combined,
        [sample_size, sample_size, num_in_channels + num_out_channels])
    input_img = combined[:, :, 0:num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img


def center_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    central_fraction = sample_size / input_img.shape[0]
    input_img = tf.image.central_crop(input_img, central_fraction)
    output_img = tf.image.central_crop(output_img, central_fraction)
    return input_img, output_img

# --------------------------
# Dataset Reader
# --------------------------

def _get_base_key(key: Text) -> Text:
    match = re.match(r'[a-zA-Z]+', key)
    if match:
        return match.group(1)
    raise ValueError(f'The provided key does not match the expected pattern: {key}')


def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(f'No data statistics available for the requested key: {key}.')
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(f'No data statistics available for the requested key: {key}.')
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)


def _get_features_dict(sample_size: int, features: List[Text]) -> Dict[Text, tf.io.FixedLenFeature]:
    sample_shape = [sample_size, sample_size]
    return {
        key: tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32, default_value=[0.0] * (sample_size * sample_size))
        for key in features
    }


def _parse_fn(example_proto: tf.train.Example, data_size: int, sample_size: int,
              num_in_channels: int, clip_and_normalize: bool, clip_and_rescale: bool,
              random_crop: bool, center_crop: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    try:
        input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
        feature_names = input_features + output_features
        features_dict = _get_features_dict(data_size, feature_names)
        features = tf.io.parse_single_example(example_proto, features_dict)
    except tf.errors.InvalidArgumentError as e:
        #invalid example, doesn't include some feature, found this in practice to usually be PrevFireMask
        #will filter out at a later point
        return tf.constant(-999.0, shape=[1]), tf.constant(-999.0, shape=[1])

    if clip_and_normalize:
        inputs_list = [_clip_and_normalize(features.get(key), key) for key in input_features]
    elif clip_and_rescale:
        inputs_list = [_clip_and_rescale(features.get(key), key) for key in input_features]
    else:
        inputs_list = [features.get(key) for key in input_features]

    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, 'outputs_list should not be empty'
    outputs_stacked = tf.stack(outputs_list, axis=0)

    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, (
        f'outputs_stacked should be rank 3 but got {outputs_stacked_shape}'
    )
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if 'FireMask' in output_features:
        output_img = map_fire_labels(output_img)

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(
            input_img, output_img, sample_size, num_in_channels, 1)
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(
            input_img, output_img, sample_size)

    return input_img, output_img

def is_valid_sample(x, y):
    return tf.reduce_all(x > -999)

def get_dataset(file_pattern: Text, data_size: int, sample_size: int,
                batch_size: int, num_in_channels: int, compression_type: Text,
                clip_and_normalize: bool, clip_and_rescale: bool,
                random_crop: bool, center_crop: bool) -> tf.data.Dataset:
    if clip_and_normalize and clip_and_rescale:
        raise ValueError('Cannot have both normalize and rescale.')

    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_fn(
            x, data_size, sample_size, num_in_channels,
            clip_and_normalize, clip_and_rescale, random_crop, center_crop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #filter out invalid examples
    dataset = dataset.filter(is_valid_sample)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

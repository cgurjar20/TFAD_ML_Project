# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import warnings
import time
from pathlib import Path, PosixPath
from typing import Optional, Union
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import tfad
from tfad.ts import TimeSeries, TimeSeriesDataset
from tfad.ts import transforms as tr
from tfad.model import TFAD, TFADDataModule


def nasa_inject_anomalies(
    dataset: TimeSeriesDataset,
    injection_method: str = ["None", "local_outliers", "smap_specialized"][-2],
    ratio_injected_spikes: float = None,
) -> TimeSeriesDataset:

    if injection_method == "None":
        return dataset
    elif injection_method == "local_outliers":
        if ratio_injected_spikes is None:
            ts_transform = tr.LocalOutlier(area_radius=40, num_spikes=10)
        else:
            ts_transform = tr.LocalOutlier(
                area_radius=500,
                num_spikes=ratio_injected_spikes,
                spike_multiplier_range=(1.0, 3.0),
            )

        multiplier = 10
        ts_transform_iterator = ts_transform(itertools.cycle(dataset))
        dataset_transformed = tfad.utils.take_n_cycle(
            ts_transform_iterator, multiplier * len(dataset)
        )
        dataset_transformed = TimeSeriesDataset(dataset_transformed)

    elif injection_method == "smap_specialized":
        ts_transform = tr.smap_injection()
        multiplier = 10
        ts_transform_iterator = ts_transform(itertools.cycle(dataset))
        dataset_transformed = tfad.utils.take_n_cycle(
            ts_transform_iterator, multiplier * len(dataset)
        )
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    else:
        raise ValueError(f"injection_method = {injection_method} not supported!")

    return dataset_transformed


def rolling_fft_features(values: np.ndarray, fft_window: int = 64) -> np.ndarray:
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    T, C = values.shape
    half = fft_window // 2
    feats = np.zeros((T, 4 * C), dtype=np.float32)

    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half)

        w = values[start:end]
        if len(w) < 4:
            w = values[max(0, t - 2):min(T, t + 2)]

        for c in range(C):
            x = w[:, c].astype(np.float32)

            if len(x) < 2:
                continue

            fft_vals = np.fft.rfft(x)
            fft_mag = np.abs(fft_vals)

            mean_fft = float(np.mean(fft_mag))
            std_fft = float(np.std(fft_mag))
            dom_idx = float(np.argmax(fft_mag))
            energy = float(np.sum(fft_mag ** 2))

            dom_norm = dom_idx / max(1, len(fft_mag) - 1)

            base = 4 * c
            feats[t, base + 0] = mean_fft
            feats[t, base + 1] = std_fft
            feats[t, base + 2] = dom_norm
            feats[t, base + 3] = energy

    return feats


def augment_dataset_with_local_fft_features(
    dataset: TimeSeriesDataset,
    fft_window: int = 64
) -> TimeSeriesDataset:
    augmented = TimeSeriesDataset()

    for ts in dataset:
        values = ts.values
        labels = ts.labels

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        fft_feats = rolling_fft_features(values, fft_window=fft_window)

        mean = np.mean(fft_feats, axis=0, keepdims=True)
        std = np.std(fft_feats, axis=0, keepdims=True) + 1e-8
        fft_feats = (fft_feats - mean) / std

        augmented_values = np.concatenate(
            [values.astype(np.float32), fft_feats.astype(np.float32)],
            axis=1
        )

        augmented.append(
            TimeSeries(
                values=augmented_values,
                labels=labels
            )
        )

    return augmented


def nasa_pipeline(
    data_dir: Union[str, PosixPath],
    model_dir: Union[str, PosixPath],
    log_dir: Union[str, PosixPath],
    benchmark: str = ["SMAP", "MSL"][0],
    exp_name: Optional[str] = None,
    epochs: int = 500,
    gpus: int = 1 if torch.cuda.is_available() else 0,
    injection_method: str = ["None", "local_outliers", "smap_specialized"][-2],
    ratio_injected_spikes: float = None,
    window_length: int = 400,
    suspect_window_length: int = 10,
    num_series_in_train_batch: int = 16,
    num_crops_per_series: int = 16,
    num_workers_loader: int = 0,
    fft_window: int = 64,
    tcn_kernel_size: int = 5,
    tcn_layers: int = 4,
    tcn_out_channels: int = 32,
    tcn_maxpool_out_channels: int = 32,
    embedding_rep_dim: int = 128,
    normalize_embedding: bool = True,
    distance: str = ["cosine", "L2", "non-contrastive"][0],
    classifier_threshold: float = 0.5,
    threshold_grid_length_test: float = 0.05,
    coe_rate: float = 0.5,
    mixup_rate: float = 2.0,
    learning_rate: float = 3e-4,
    stride_roll_pred_val_test: int = 2,
    test_labels_adj: bool = True,
    max_windows_unfold_batch: Optional[int] = 5000,
    evaluation_result_path: Optional[Union[str, PosixPath]] = None,
    rnd_seed: int = 123,
    **kwargs,
):

    dirs = [data_dir, model_dir, log_dir]
    data_dir, model_dir, log_dir = [
        PosixPath(path).expanduser() if str(path).startswith("~") else Path(path)
        for path in dirs
    ]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if (not os.path.exists(log_dir)) and (not str(log_dir).startswith("s3://")):
        os.makedirs(log_dir)

    pl.trainer.seed_everything(rnd_seed)

    train_set, test_set = tfad.datasets.nasa(
        path=data_dir,
        benchmark=benchmark,
    )

    univariate = False
    if univariate:
        N = len(train_set)
        for i in range(N):
            train_set[i].values = train_set[i].values[:, 0]
            test_set[i].values = test_set[i].values[:, 0]

    scaler = tr.TimeSeriesScaler(type="robust")
    train_set = TimeSeriesDataset(tfad.utils.take_n_cycle(scaler(train_set), len(train_set)))
    test_set = TimeSeriesDataset(tfad.utils.take_n_cycle(scaler(test_set), len(test_set)))

    train_set = augment_dataset_with_local_fft_features(train_set, fft_window=fft_window)
    test_set = augment_dataset_with_local_fft_features(test_set, fft_window=fft_window)

    ts_channels = train_set[0].shape[1]
    assert all(shape[1] == ts_channels for shape in train_set.shape)
    assert all(shape[1] == ts_channels for shape in test_set.shape)
    print("Total channels after adding local FFT features:", ts_channels)

    split_idx = len(test_set) // 2
    validation_set = TimeSeriesDataset(test_set[:split_idx])
    test_set = TimeSeriesDataset(test_set[split_idx:])

    print("Validation series count:", len(validation_set))
    print("Test series count:", len(test_set))

    for i, ts in enumerate(validation_set):
        print(f"Validation set series {i} positive labels:", int(ts.labels.sum()))

    for i, ts in enumerate(test_set):
        print(f"Test set series {i} positive labels:", int(ts.labels.sum()))

    for i, ts in enumerate(train_set):
        if ts.shape[0] < window_length:
            pad_len = (window_length - ts.shape[0]) + num_crops_per_series
            assert pad_len > 0
            warnings.warn(
                f"Padding series {i} with {pad_len} timesteps on the left "
                f"({np.round(100 * pad_len / ts.shape[0], 0)} % of its original length)"
            )
            ts_pad = TimeSeries(
                values=np.zeros((pad_len, ts.shape[1])).squeeze(),
                labels=np.zeros(pad_len)
            )
            train_set[i] = ts_pad.append(train_set[i])

    train_set_transformed = nasa_inject_anomalies(
        dataset=train_set,
        injection_method=injection_method,
        ratio_injected_spikes=ratio_injected_spikes,
    )

    data_module = TFADDataModule(
        train_ts_dataset=train_set_transformed,
        validation_ts_dataset=validation_set,
        test_ts_dataset=test_set,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        num_series_in_train_batch=num_series_in_train_batch,
        num_crops_per_series=num_crops_per_series,
        label_reduction_method="any",
        stride_val_and_test=stride_roll_pred_val_test,
        num_workers=num_workers_loader,
    )

    if distance == "cosine":
        distance = tfad.model.distances.CosineDistance()
    elif distance == "L2":
        distance = tfad.model.distances.LpDistance(p=2)
    elif distance == "non-contrastive":
        distance = tfad.model.distances.BinaryOnX1(rep_dim=embedding_rep_dim, layers=1)

    model = TFAD(
        ts_channels=ts_channels,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        tcn_kernel_size=tcn_kernel_size,
        tcn_layers=tcn_layers,
        tcn_out_channels=tcn_out_channels,
        tcn_maxpool_out_channels=tcn_maxpool_out_channels,
        embedding_rep_dim=embedding_rep_dim,
        normalize_embedding=normalize_embedding,
        distance=distance,
        classification_loss=nn.BCELoss(),
        classifier_threshold=classifier_threshold,
        threshold_grid_length_test=threshold_grid_length_test,
        coe_rate=coe_rate,
        mixup_rate=mixup_rate,
        stride_rolling_val_test=stride_roll_pred_val_test,
        val_labels_adj=True,
        test_labels_adj=test_labels_adj,
        max_windows_unfold_batch=max_windows_unfold_batch,
        learning_rate=learning_rate,
    )

    if exp_name is None:
        time_now = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
        exp_name = f"nasa-{time_now}"

    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)

    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss_step",
        dirpath=model_dir,
        filename="tfad-model-" + exp_name + "-{epoch:02d}-{train_loss_step:.4f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=model_dir,
        logger=logger,
        min_epochs=epochs,
        max_epochs=epochs,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    ckpt_path = checkpoint_cb.best_model_path

    if ckpt_path:
        print(f"Checkpoint being loaded: {ckpt_path}")
        model = TFAD.load_from_checkpoint(ckpt_path, weights_only=False)
    else:
        print(f"No checkpoint saved in {model_dir}. Using current in-memory model.")

    print("Loaded checkpoint threshold:", model.hparams.classifier_threshold)

    thresholds_to_try = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    best_thr = None
    best_f1 = -1.0

    print("\nValidation threshold sweep:\n")

    original_test_set = data_module.test_ts_dataset
    data_module.test_ts_dataset = data_module.validation_ts_dataset

    for thr in thresholds_to_try:
        model.hparams.classifier_threshold = thr
        result_thr = trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=None,
            verbose=False
        )[0]

        print(
            f"threshold={thr:.2f} | "
            f"val_f1={result_thr['test_f1']:.6f} | "
            f"val_precision={result_thr['test_precision']:.6f} | "
            f"val_recall={result_thr['test_recall']:.6f} | "
            f"val_TP={result_thr['test_TP']} | "
            f"val_FP={result_thr['test_FP']} | "
            f"val_TN={result_thr['test_TN']} | "
            f"val_FN={result_thr['test_FN']}"
        )

        if result_thr["test_f1"] > best_f1:
            best_f1 = result_thr["test_f1"]
            best_thr = thr

    data_module.test_ts_dataset = original_test_set
    model.hparams.classifier_threshold = best_thr

    print(f"\nBest threshold selected from validation: {best_thr:.2f}")
    print(f"Best validation F1: {best_f1:.6f}")

    evaluation_result = trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=None
    )[0]

    evaluation_result["selected_threshold"] = best_thr

    print("\nFinal test evaluation using learned threshold:\n")
    for key, value in evaluation_result.items():
        print(f"{key}={value}")

    f1 = float(evaluation_result["test_f1"])
    precision = float(evaluation_result["test_precision"])
    recall = float(evaluation_result["test_recall"])
    f2 = float(evaluation_result["test_f2"])
    f05 = float(evaluation_result["test_f0.5"])

    tn = int(evaluation_result["test_TN"])
    fn = int(evaluation_result["test_FN"])
    tp = int(evaluation_result["test_TP"])
    fp = int(evaluation_result["test_FP"])
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else 0.0

    evaluation_result["test_accuracy"] = accuracy
    evaluation_result["test_specificity"] = specificity
    evaluation_result["test_balanced_accuracy"] = balanced_accuracy
    evaluation_result["test_false_positive_rate"] = false_positive_rate
    evaluation_result["test_false_negative_rate"] = false_negative_rate
    evaluation_result["test_mcc"] = mcc

    print(f"test_accuracy={accuracy}")
    print(f"test_specificity={specificity}")
    print(f"test_balanced_accuracy={balanced_accuracy}")
    print(f"test_false_positive_rate={false_positive_rate}")
    print(f"test_false_negative_rate={false_negative_rate}")
    print(f"test_mcc={mcc}")

    if evaluation_result_path is not None:
        path = evaluation_result_path
        path = PosixPath(path).expanduser() if str(path).startswith("~") else Path(path)
        with open(path, "w") as f:
            json.dump(evaluation_result, f, cls=tfad.utils.NpEncoder)

    metrics = [
        "F1", "Precision", "Recall", "F2", "F0.5",
        "Accuracy", "Specificity", "Bal.Acc", "MCC"
    ]
    values = [
        f1, precision, recall, f2, f05,
        accuracy, specificity, balanced_accuracy, mcc
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"TFAD Performance on {benchmark} with FFT Features")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(model_dir / f"{benchmark.lower()}_metrics_bar.png")
    plt.close()

    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix for {benchmark} with FFT Features")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(model_dir / f"{benchmark.lower()}_confusion_matrix.png")
    plt.close()

    labels_pie = ["TP", "FP", "TN", "FN"]
    sizes = [tp, fp, tn, fn]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels_pie, autopct="%1.1f%%", startangle=90)
    plt.title(f"Prediction Outcome Distribution on {benchmark} with FFT Features")
    plt.tight_layout()
    plt.savefig(model_dir / f"{benchmark.lower()}_prediction_pie.png")
    plt.close()

    summary_text = f"""
TFAD Results on {benchmark} with FFT Features

Threshold   : {best_thr:.2f}
F1 Score    : {f1:.3f}
Precision   : {precision:.3f}
Recall      : {recall:.3f}
F2 Score    : {f2:.3f}
F0.5 Score  : {f05:.3f}
Accuracy    : {accuracy:.3f}
Specificity : {specificity:.3f}
Bal. Acc    : {balanced_accuracy:.3f}
FPR         : {false_positive_rate:.3f}
FNR         : {false_negative_rate:.3f}
MCC         : {mcc:.3f}

TP = {tp}
FP = {fp}
TN = {tn}
FN = {fn}
"""

    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0.05, 0.95, summary_text, va="top", fontsize=12)
    plt.title(f"{benchmark} Result Summary")
    plt.tight_layout()
    plt.savefig(model_dir / f"{benchmark.lower()}_summary.png")
    plt.close()

    print("Saved visualizations in:", model_dir)
    print(f"tfad on nasa {benchmark} dataset finished successfully!")


from general_parser import get_general_parser
from tfad.utils import save_args


if __name__ == "__main__":
    parser = get_general_parser()
    parser.add_argument("--benchmark", type=str, default=["SMAP", "MSL"][0])

    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    model_dir = args_dict["model_dir"].expanduser()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_args(args=args_dict, path=model_dir / "args.json")
    nasa_pipeline(**args_dict)
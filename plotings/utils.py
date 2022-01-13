from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os

BOXPLOT_METRIC = 'Pearson Correlation'

def get_values(raw_info_dataframe: pd.DataFrame, experiment_name: str) -> dict:
    mae_avg = raw_info_dataframe['MAE'].mean(axis=0)
    mae_std = raw_info_dataframe['MAE'].std(axis=0)
    mape_avg = raw_info_dataframe['MAPE'].mean(axis=0)
    mape_std = raw_info_dataframe['MAPE'].std(axis=0)
    correlation_avg = raw_info_dataframe['Pearson Correlation'].mean(axis=0)
    correlation_std = raw_info_dataframe['Pearson Correlation'].std(axis=0)
    values_list = [mae_avg, mae_std, mape_avg,
                   mape_std, correlation_avg, correlation_std]
    values = {experiment_name: values_list}
    return values


def get_all_experiment_names(experiment_folder: str) -> list:
    exps_names = []
    for experiment in os.listdir(experiment_folder):
        for root, dirs, files in os.walk(experiment_folder+experiment):
            for file_name in files:
                if 'fold_metrics.csv' in file_name:
                    exps_names.append(experiment)
    exps_names.sort()
    return exps_names


def get_all_experiments_values(experiment_folder: str, metric='MAE', metric_index=False) -> pd.DataFrame:
    df = pd.DataFrame()
    for experiment in os.listdir(experiment_folder):
        for root, dirs, files in os.walk(experiment_folder+experiment):
            for file_name in files:

                if 'fold_metrics.csv' in file_name:
                    raw_info_df = pd.read_csv(os.path.join(
                        experiment_folder, experiment, file_name), index_col=None, header=0)

                    if metric_index:
                        df_rows = pd.DataFrame(
                            data=get_values(raw_info_df, experiment))
                    else:
                        df_rows = pd.DataFrame(
                            {experiment: raw_info_df[metric]})

                    df.insert(0, experiment, df_rows[experiment])

    if metric_index:
        indexes = ['MAE', 'MAE STD', 'MAPE', 'MAPE STD',
                   'Correlation', 'Correlation STD']
        df.index = indexes
    df = df.reindex(sorted(df.columns), axis=1)

    return df


def boxplot(path):
    data = get_all_experiments_values(path, BOXPLOT_METRIC)
    fig1, ax1 = plt.subplots()
    ax1.boxplot(data)
    ax1.set_ylabel(BOXPLOT_METRIC)
    ax1.yaxis.grid(True)

    plt.xticks(rotation=270)
    plt.setp(ax1, xticklabels=get_all_experiment_names(path))
    plt.tight_layout()

    if not os.path.isdir('boxplot'):
        os.makedirs("boxplot")

    plt.savefig('boxplot/' + 'box_plot.pdf')
    plt.close()


def histogram(path):
    for experiment in os.listdir(path):
        for root, dirs, files in os.walk(path+experiment):
            for file_name in files:
                if 'predictions.csv' in file_name:
                    data = pd.read_csv(os.path.join(
                        path, experiment, "predictions.csv"))
                    plot_and_save_histogram(
                        experiment, data["real_value"].values, data["prediction"].values, 20)


def scatter(path):
    for experiment in os.listdir(path):
        for root, dirs, files in os.walk(path+experiment):
            for file_name in files:
                if 'predictions.csv' in file_name:
                    data = pd.read_csv(os.path.join(
                        path, experiment, "predictions.csv"))
                    scatter_plot_and_save(
                        experiment, data["real_value"].values, data["prediction"].values)


def loss(path):
    df, aliases = get_all_experiments_avg_validation_loss(
        path, show_model_names=True)
    df = df[["epoch", "validation_loss", "model"]]
    df.set_index("epoch", inplace=True)
    df = df.pivot(columns="model")
    df["validation_loss"].plot()
    plt.ylabel("Validation Loss")
    plt.legend(title="")
    plt.xlabel("Epochs")

    if not os.path.isdir("loss"):
        os.makedirs("loss")

    plt.tight_layout()
    plt.savefig("loss/" + "overall_loss.pdf")
    plt.close()


def calc_intersections(hist1, hist2) -> float:
    s = 0
    for p in zip(hist1, hist2):
        s += min(p)

    return s


def plot_and_save_histogram(experiment_name: str,
                            real: Tuple,
                            pred: Tuple,
                            bins: int,
                            weights=None) -> None:
    plt.figure()

    weights = np.ones(len(real)) / len(real)

    range_min = min(np.min(real), np.min(pred))
    range_max = max(np.max(real), np.max(pred))

    full_range = (range_min, range_max)

    n1, bins, _ = plt.hist(real, bins=bins,
                           range=full_range,
                           weights=weights,
                           facecolor="#34a2eb",
                           edgecolor="#2c5aa3",
                           alpha=0.9)

    n2, bins, _ = plt.hist(pred, bins=bins,
                           range=full_range,
                           weights=weights,
                           facecolor="#ffbc47",
                           #    edgecolor="#9e742b",
                           alpha=0.6)

    real_patch = mpatches.Patch(color='#34a2eb', label='y')
    pred_patch = mpatches.Patch(color='#ffbc47', label='ŷ')
    plt.legend(handles=[real_patch, pred_patch])

    intersection = calc_intersections(n1, n2)
    plt.tight_layout()

    if not os.path.isdir("histograms"):
        os.makedirs("histograms")

    plt.savefig("histograms/" + experiment_name +
                "_" + str(intersection)[:4:] + ".pdf")
    plt.close()


def scatter_plot_and_save(experiment_name: str,
                          real: Tuple,
                          pred: Tuple) -> None:

    plt.figure()
    plt.xlabel("REAL")
    plt.ylabel("PREDICTION")

    plt.plot(real, pred, 'co')
    dashes = [5, 5, 5, 5]

    plt.plot(real, real, dashes=dashes, color="#cccccc")
    plt.tight_layout()

    if not os.path.isdir("scatter_graphs"):
        os.makedirs("scatter_graphs")

    plt.savefig("scatter_graphs/" + experiment_name + "_" + "scatter.pdf")
    plt.close()


def get_all_experiments_avg_validation_loss(experiment_folder: str, show_model_names: bool = True):

    experiments_df = pd.DataFrame(
        columns=["epoch", "train_loss", "validation_loss", "model"])

    for experiment in os.listdir(experiment_folder):
        for root, dirs, files in os.walk(experiment_folder+experiment):
            for file_name in files:

                if "raw_fold_info.csv" in file_name:

                    raw_info_df = pd.read_csv(os.path.join(
                        experiment_folder, experiment, file_name))
                    experiment_avg_loss = get_avg_validation_loss(
                        raw_info_df).reset_index()
                    experiment_avg_loss["model"] = experiment

                    experiments_df = experiments_df.append(
                        experiment_avg_loss, ignore_index=True)

    models = experiments_df["model"].unique()
    models.sort()
    aliases = {name: f"#{i}" for i, name in enumerate(models, start=1)}

    if not show_model_names:
        experiments_df.replace(aliases, inplace=True)

    return experiments_df, aliases


def get_avg_validation_loss(raw_fold_info_df: pd.DataFrame) -> pd.DataFrame:

    folds = len(raw_fold_info_df["fold"].unique())

    new_df = raw_fold_info_df[["epoch", "train_loss",
                               "validation_loss"]].groupby("epoch").sum()
    new_df["train_loss"] /= folds
    new_df["validation_loss"] /= folds

    return new_df
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels


def summarize_and_save_model_report(preds, labels, name):
    os.makedirs("results", exist_ok=True)
    summary = metrics.classification_report(labels, preds, output_dict=True)
    confusion_matrix = metrics.confusion_matrix(labels, preds)
    summary.update({
        "preds": preds.tolist(),
        "labels": labels.tolist(),
    })
    display_labels = unique_labels(labels, preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    cm_display.plot(
        include_values=True,
        cmap="viridis",
        ax=None,
        xticks_rotation="horizontal",
        values_format=None,
        colorbar=True,
    )
    plt.title(name.upper())
    plt.savefig(f'results/{name}-confusion-matrix.png')
    plt.show()

    with open(f"results/{name}-summary.yaml", "w") as f:
        yaml.dump(summary, f)
    print(f"Dumped model summary at: {name}-summary.yaml and confusion matrix at: {name}-confusion-matrix.png")


def visualize_per_class(models, summaries, name):
    plt.clf()
    classes = ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]
    for summary, model, color in zip(summaries, models, ["b", "g", "r", "c"]):
        precisions = [float(summary[_class][name]) for _class in classes]
        plt.plot(classes, precisions, color, label=model.upper(), linewidth=2)
    plt.legend()
    plt.grid()
    plt.xlabel('Classes')
    plt.ylabel(name.capitalize())
    plt.title(f'{name.capitalize()} Class plots')
    plt.savefig(f"results/{name}.png")
    plt.show()
    print(f"Dumped {name.capitalize()} plot at results/{name}.png")


def visualize_per_model_acc(models, summaries):
    plt.clf()
    accs = [float(summary["accuracy"]) for summary in summaries]
    plt.plot(models, accs, linewidth=2)
    plt.grid()
    plt.ylabel('Accuracy')
    plt.xlabel("Models")
    plt.title(f'Accuracy plots')
    plt.savefig(f"results/accuracy.png")
    plt.show()
    print(f"Dumped Accuracy plot at results/accuracy.png")


if __name__ == "__main__":
    models = [
        "knn",
        "svm",
        "mlp",
        "adaboost_tree"
    ]
    summaries = [yaml.load(open(f"results/{m}-summary.yaml"), Loader=yaml.BaseLoader) for m in models]

    # plot and save precision, f1, recall
    visualize_per_class(models, summaries, "precision")
    visualize_per_class(models, summaries, "recall")
    visualize_per_class(models, summaries, "f1-score")

    # plot accuracies
    visualize_per_model_acc(models, summaries)

    # todo: plot roc characteristics


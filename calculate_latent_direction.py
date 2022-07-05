import os
from pathlib import Path
from sklearn import svm

import numpy as np

def remove_suffix(input_path):
    files_in_path = input_path.glob("*")
    return [n.stem for n in files_in_path]

def get_input_latent(source_path, relative_label_path):
    latents = []

    labels = remove_suffix(source_path / relative_label_path)
    for latent_path in (source_path / "latents").glob("*.npz"):
        if latent_path.stem in labels:
            w = np.load(latent_path)['w']
            latents.append(w)

    latents = np.vstack(latents)
    return latents
    

def calculate_latent_direction(source_path):
    source_path = Path(source_path)
    positives = get_input_latent(source_path, "positive")
    negatives = get_input_latent(source_path, "negative")

    train_data = np.vstack([positives, negatives])
    train_label = np.concatenate([np.ones(len(positives), dtype=np.int),
                                  np.zeros(len(negatives), dtype=np.int)])

    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)
    seperator = classifier.coef_[0].astype(np.float32)
    seperator = seperator / np.linalg.norm(seperator)

    np.savez(source_path / "seperator.npz", sep=seperator)

if __name__ == "__main__":
    calculate_latent_direction("output/cat_ffhq_pose")
"""Export verified numeric GRU weights for the local browser worker."""

import json

import h5py

from setup_tennis_rnn import ROOT, ARTIFACTS, main as setup


def main():
    setup()
    paths = {
        "kernel": "gru/gru/gru_cell/kernel:0",
        "recurrent": "gru/gru/gru_cell/recurrent_kernel:0",
        "bias": "gru/gru/gru_cell/bias:0",
        "denseKernel": "dense/dense/kernel:0", "denseBias": "dense/dense/bias:0",
        "outputKernel": "dense_1/dense_1/kernel:0", "outputBias": "dense_1/dense_1/bias:0",
    }
    with h5py.File(ROOT / "tennis_rnn.h5", "r") as handle:
        weights = {name: handle["model_weights"][path][:].reshape(-1).tolist() for name, path in paths.items()}
    payload = {"version": 1, "checkpointSHA256": ARTIFACTS["tennis_rnn.h5"][1],
               "classes": ["backhand", "forehand", "neutral", "serve"], "weights": weights}
    destination = ROOT / "weights.json"
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Exported numeric browser weights: {destination}")


if __name__ == "__main__":
    main()

# HGSF: Hybrid GraphSAGE with Structural Feature Fusion for Link Prediction

Code for the paper **"HGSF: Gated Fusion of GraphSAGE and Structural Features for Link Prediction, Evaluated with Random and Distance-Matched Negatives"** (Karwan Mohammad Hamakarim, Zana Othman Hamad, Aziz Rahman, Shoxan Abdula; 2026).

HGSF scores a candidate node pair by fusing (i) a GraphSAGE pair representation with (ii) ten explicit
structural similarity features (CN, AA, JC, RA, PA, Salton, HPI, L3, minimum degree, maximum degree)
through a learned gate. This repository contains the complete, leakage-controlled evaluation pipeline used in the paper.

## What is compared

All methods use the same edge splits, the same negative samples, and the same threshold rule.

| Group | Methods |
|---|---|
| Classical heuristics | CN, AA, JC, RA, PA |
| Learned, structural features only | StructMLP (MLP on the 10 structural features) |
| Learned, GraphSAGE only | GraphSAGE |
| Learned, fusion by concatenation | SAGE+Struct (concat) |
| Proposed | HGSF (gated fusion) |

## Evaluation protocol

* Edges are split 85 / 5 / 10 (train / validation / test) with a new random split for every seed.
* Validation and test edges are removed from the graph **before** any feature or embedding is computed.
* For training pairs, the target edge is removed when computing structural features.
* In every training epoch, 30% (default) of the training edges are held out from message passing and used as supervision.
* Negatives: **random**, or **distance-matched hard negatives** (each negative has the same shortest-path class
  2 / 3 / 4+ hops as a real positive pair), which removes the common-neighbor shortcut of two-hop sampling.
* The decision threshold (for Precision, Recall, F1) is chosen on the validation set only.
* Hyperparameters (hidden size, learning rate, dropout, supervision fraction) are tuned on validation AUC only (`--tune`).
* Metrics: AUC, AP, Precision, Recall, F1; mean and standard deviation over seeds; paired t-tests of HGSF against every other method.

## Installation

```bash
pip install torch torch_geometric scikit-learn scipy numpy
```

The code was run on Google Colab with a GPU (a CPU also works, but is much slower).

## Data

* **Facebook**: download `facebook_combined.txt.gz` from https://snap.stanford.edu/data/ego-Facebook.html
* **Cora, CiteSeer, PubMed**: downloaded automatically by PyTorch Geometric (`Planetoid`) into `./data`.

## Usage

```bash
# Cora, 5 seeds, random and distance-matched negatives, validation-based tuning
python hgsf_all.py --dataset cora --seeds 5 --neg both --tune

# Facebook
python hgsf_all.py --dataset facebook --edges facebook_combined.txt.gz --seeds 5 --neg both --tune

# CiteSeer and PubMed (PubMed without tuning, as in the paper)
python hgsf_all.py --dataset citeseer --seeds 5 --neg both --tune
python hgsf_all.py --dataset pubmed --seeds 5 --neg both
```

In the output, `random` means uniform random negatives and `hard` means distance-matched hard negatives.

## Additional experiments (ablation and timing)

`hgsf_extra.py` (needs `hgsf_all.py` in the same folder) runs two extra studies:

```bash
# Feature ablation: HGSF with 5 classic features / 9 features (no L3) / all 10 features
python hgsf_extra.py --task ablation --dataset cora --seeds 5 --neg both

# Parameters, training time, inference time, and feature-computation time
python hgsf_extra.py --task timing --dataset cora --seeds 3
```

All ablation variants use the same default hyperparameters (hidden size 64, learning rate 0.005,
dropout 0.3, supervision fraction 0.3) and the same splits and negatives. Timing numbers depend on the
hardware (GPU or CPU) used for the run.

### Timing (cleaner version) and two-hop negatives

```bash
# Seconds per epoch (fixed 20 epochs after warm-up), parameters, inference time, feature time
python hgsf_timing2.py --dataset cora --seeds 3

# Common two-hop negatives (used only to show the common-neighbour shortcut, Appendix B of the paper)
python hgsf_twohop.py --dataset cora --seeds 5 --neg random
```

`hgsf_timing2.py` replaces the timing task of `hgsf_extra.py` (it does not depend on early stopping).
`hgsf_twohop.py` draws every negative pair exactly two hops apart; the results are saved with the tag
`<dataset>_twohop_v4_random`.

## Output files

For every dataset and negative protocol the script writes (tag: `<dataset>_v4_<negatives>`):

* `results_<tag>.csv`: mean ± std of all metrics for all methods (the tables of the paper);
* `seeds_<tag>.csv`: the result of every individual seed (used for the paired t-tests);
* the console shows the tuned configurations, the mean gate value, and the t-test p-values.

## Notes

* For graphs where most heuristics are close to chance (Cora, CiteSeer, PubMed with distance-matched negatives),
  the F1 of a heuristic can collapse to 0.667 (all pairs predicted as links); AUC and AP are the primary metrics.
* Results can vary slightly between GPU types and library versions.

## Citation

```
The BibTeX entry will be added after publication.
```

## License

MIT License (see the LICENSE file).

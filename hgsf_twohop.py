"""
Runs hgsf_all.py with the common TWO-HOP negatives (every negative pair is exactly two hops apart),
so that the common-neighbour shortcut can be shown in a table (Appendix B of the paper).
Usage:  python hgsf_twohop.py --dataset cora --seeds 5 --neg random
(the flag --neg random is used only to switch on this sampler; results are saved with the tag *_twohop_*)
"""
import copy

import hgsf_all as H

_orig_sample_neg = H.sample_neg
_orig_run_setting = H.run_setting


def sample_neg_twohop(n, k, exclude, rng, adj_list, hard):
    return _orig_sample_neg(n, k, exclude, rng, adj_list, True)


def run_setting_twohop(edges, n, base, neg, args, dev):
    a = copy.copy(args)
    a.dataset = args.dataset + "_twohop"
    return _orig_run_setting(edges, n, base, neg, a, dev)


H.sample_neg = sample_neg_twohop
H.run_setting = run_setting_twohop

if __name__ == "__main__":
    H.main()

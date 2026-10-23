## Test metrics

| model | n_features | ndcg@10 | ndcg@100 | recall@10 | recall@100 | auc | logloss | val_ndcg@10 | trees | fit_rank_s |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate scores only | 0 | 0.1966 | 0.2356 | 0.1960 | 0.3175 | 0.8145 | 0.1104 | 0.1892 | 111 | 3.0655 |
| manual | 41 | 0.2657 | 0.2834 | 0.2595 | 0.3486 | 0.8893 | 0.0976 | 0.2596 | 450 | 9.4517 |
| auto: top-B by importance | 41 | 0.2684 | 0.2844 | 0.2622 | 0.3471 | 0.8862 | 0.0979 | 0.2607 | 176 | 5.7093 |
| auto: forward (budget B) | 14 | 0.2631 | 0.2811 | 0.2572 | 0.3448 | 0.8798 | 0.0993 | 0.2554 | 192 | 5.0687 |
| auto: forward | 14 | 0.2631 | 0.2811 | 0.2572 | 0.3448 | 0.8798 | 0.0993 | 0.2554 | 192 | 5.1057 |
| auto: forward + window refine | 14 | 0.2635 | 0.2812 | 0.2576 | 0.3445 | 0.8801 | 0.0992 | 0.2580 | 121 | 4.4603 |
| auto: all screened | 232 | 0.2688 | 0.2849 | 0.2627 | 0.3478 | 0.8876 | 0.0976 | 0.2619 | 156 | 14.6102 |
| manual + auto (fwd+win) | 54 | 0.2708 | 0.2865 | 0.2642 | 0.3497 | 0.8928 | 0.0967 | 0.2635 | 149 | 5.6769 |
| heur: repeat (rep_rank) | None | 0.1861 | 0.2294 | 0.1872 | 0.3148 | None | None | None | None | None |
| heur: item2item score | None | 0.0579 | 0.0773 | 0.0610 | 0.1194 | None | None | None | None | None |
| heur: popularity 1d | None | 0.0260 | 0.0969 | 0.0275 | 0.2047 | None | None | None | None | None |
| heur: rep > i2i > artist > pop | None | 0.1749 | 0.2247 | 0.1796 | 0.3159 | None | None | None | None | None |

## Ablation (manual set minus a group)

| model | n_features | ndcg@10 | recall@100 |
|---|---|---|---|
| manual | 41 | 0.2657 | 0.3486 |
| manual - user | 33 | 0.2601 | 0.3460 |
| manual - item | 30 | 0.2645 | 0.3458 |
| manual - artist | 39 | 0.2657 | 0.3478 |
| manual - user-item (direct) | 31 | 0.2408 | 0.3425 |
| manual - user-artist/album (indirect) | 31 | 0.2639 | 0.3454 |

## Paired bootstrap, test NDCG@10 vs manual

| model | diff | ci95_lo | ci95_hi |
|---|---|---|---|
| candidate scores only | -0.06909 | -0.07064 | -0.06753 |
| manual | 0.00000 | 0.00000 | 0.00000 |
| auto: top-B by importance | 0.00268 | 0.00145 | 0.00347 |
| auto: forward (budget B) | -0.00255 | -0.00392 | -0.00164 |
| auto: forward | -0.00255 | -0.00392 | -0.00164 |
| auto: forward + window refine | -0.00214 | -0.00338 | -0.00131 |
| auto: all screened | 0.00312 | 0.00200 | 0.00390 |
| manual + auto (fwd+win) | 0.00509 | 0.00430 | 0.00580 |
| heur: repeat (rep_rank) | -0.07954 | -0.08137 | -0.07762 |
| heur: item2item score | -0.20776 | -0.21042 | -0.20462 |
| heur: popularity 1d | -0.23964 | -0.24227 | -0.23695 |
| heur: rep > i2i > artist > pop | -0.09082 | -0.09238 | -0.08918 |

## Composition of `auto_fwd_win` (14 features)

| group | len |
|---|---|
| user-item (direct) | 13 |
| user-artist (indirect) | 1 |

| kind | len |
|---|---|
| share | 6 |
| cnt | 3 |
| rate | 2 |
| since_last | 1 |
| nuniq | 1 |
| since_first | 1 |

| window | len |
|---|---|
| 90d | 3 |
| 14d | 3 |
| - | 2 |
| 7d | 2 |
| 3d | 2 |
| all | 1 |
| 30d | 1 |

Selected features: `ui.since_last.lplus`, `ui.share.lplus.7d`, `ui.rate.lplus/listen.90d`, `ui.cnt.lplus.7d`, `ui.share.lplus.90d`, `ui.share.lplus.14d`, `ui.since_first.like`, `ui.cnt.listen.3d`, `ui.cnt.org.14d`, `ui.share.like.90d`, `ui.share.lplus.3d`, `ui.rate.skip/listen.all`, `ui.share.listen.14d`, `ua.nuniq.lplus.30d`

Window swaps: `ui.share.lplus.all` -> `ui.share.lplus.90d` (0.2560 -> 0.2565); `ui.cnt.listen.7d` -> `ui.cnt.listen.3d` (0.2565 -> 0.2573)

## Scaling

| set | train_users | train_rows | ndcg@10 | recall@100 | fit_s |
|---|---|---|---|---|---|
| manual | 2127 | 82455 | 0.2557 | 0.3451 | 2.1053 |
| auto: forward + window refine | 2127 | 82455 | 0.2551 | 0.3425 | 1.9245 |
| manual + auto (fwd+win) | 2127 | 82455 | 0.2604 | 0.3468 | 2.3056 |
| manual | 4255 | 165536 | 0.2585 | 0.3460 | 2.2135 |
| auto: forward + window refine | 4255 | 165536 | 0.2582 | 0.3432 | 2.1037 |
| manual + auto (fwd+win) | 4255 | 165536 | 0.2630 | 0.3479 | 4.0211 |
| manual | 8511 | 331097 | 0.2628 | 0.3473 | 3.4358 |
| auto: forward + window refine | 8511 | 331097 | 0.2612 | 0.3442 | 4.0443 |
| manual + auto (fwd+win) | 8511 | 331097 | 0.2686 | 0.3491 | 3.9791 |
| manual | 17022 | 661920 | 0.2657 | 0.3486 | 10.6291 |
| auto: forward + window refine | 17022 | 661920 | 0.2635 | 0.3445 | 4.1514 |
| manual + auto (fwd+win) | 17022 | 661920 | 0.2708 | 0.3497 | 5.8115 |

## Timings

```json
{
 "stage_cands_s": 86.63743281364441,
 "features_full_space": {
  "train": {
   "split": "train",
   "tag": "all",
   "n_features": 1236,
   "rows": 751260,
   "time_global_s": 203.88887763023376,
   "time_total_s": 220.48349165916443,
   "peak_gb": 8.804496765136719
  },
  "val": {
   "split": "val",
   "tag": "all",
   "n_features": 1236,
   "rows": 1272660,
   "time_global_s": 207.42613291740417,
   "time_total_s": 219.27875804901123,
   "peak_gb": 8.928836822509766
  }
 },
 "stage_feats_s": 439.7749373912811,
 "selection": {
  "n_space": 1236,
  "n_evals": 168,
  "time_screen_s": 52.9661386013031,
  "time_importance_s": 7.64536452293396,
  "time_wrapper_s": 427.37486505508423,
  "time_total_s": 493.4345381259918
 },
 "stage_select_s": 508.8740313053131,
 "features_test_final": {
  "split": "test",
  "tag": "final",
  "n_features": 264,
  "rows": 11632225,
  "time_global_s": 15.127082824707031,
  "time_total_s": 52.75194597244263
 },
 "features_test_manual": {
  "split": "test",
  "tag": "manual",
  "n_features": 41,
  "rows": 11632225,
  "time_global_s": 11.507186889648438,
  "time_total_s": 21.958942413330078
 },
 "stage_test_feats_s": 74.73057818412781,
 "final": {
  "predict_test_s": 1008.7326803207397,
  "test_rows": 11632225,
  "neg_rate": 0.12203380933983471,
  "peak_gb": 17.788272857666016
 },
 "stage_final_s": 1527.239931344986,
 "stage_scaling_s": 169.60291075706482
}
```

## Tuning (manual set, val)

| num_leaves | min_data_in_leaf | val_ndcg@10 | trees |
|---|---|---|---|
| 31 | 50 | 0.2587 | 665 |
| 31 | 200 | 0.2583 | 341 |
| 63 | 50 | 0.2596 | 450 |
| 63 | 200 | 0.2594 | 419 |
| 127 | 50 | 0.2592 | 233 |
| 127 | 200 | 0.2583 | 134 |

## Candidates

```json
{
 "train": {
  "n_users": 20000,
  "n_pos": 537724,
  "rows": 5066700,
  "pos_in_cands": 151431,
  "rows_saved": 751260,
  "by_source": {
   "rep_rank": [
    1825852,
    118852
   ],
   "i2i_rank": [
    1967819,
    39265
   ],
   "art_rank": [
    1181897,
    40723
   ],
   "pop_rank": [
    600000,
    9649
   ]
  },
  "cand_recall": 0.281614731721106,
  "avg_cands": 253.335,
  "time_global_s": 19.580848693847656,
  "time_total_s": 26.731685638427734
 },
 "val": {
  "n_users": 5000,
  "n_pos": 138014,
  "rows": 1272660,
  "pos_in_cands": 37728,
  "rows_saved": 1272660,
  "by_source": {
   "rep_rank": [
    459299,
    29366
   ],
   "i2i_rank": [
    493107,
    9736
   ],
   "art_rank": [
    296057,
    10078
   ],
   "pop_rank": [
    150000,
    2423
   ]
  },
  "cand_recall": 0.27336357181155535,
  "avg_cands": 254.532,
  "time_global_s": 20.667798280715942,
  "time_total_s": 25.9631826877594
 },
 "test": {
  "n_users": 45940,
  "n_pos": 1242450,
  "rows": 11632225,
  "pos_in_cands": 349592,
  "rows_saved": 11632225,
  "by_source": {
   "rep_rank": [
    4187234,
    273999
   ],
   "i2i_rank": [
    4515356,
    91049
   ],
   "art_rank": [
    2712226,
    94486
   ],
   "pop_rank": [
    1378200,
    23075
   ]
  },
  "cand_recall": 0.2813730934846473,
  "avg_cands": 253.20472355245974,
  "time_global_s": 22.66892981529236,
  "time_total_s": 33.93715500831604
 }
}
```
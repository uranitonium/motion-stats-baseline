# Third-party notices

- `hbabel/align_hbabel.py` contains the labelling loop of `datasets/hBABEL/prepare_hbabel_labels.py` (function
  `convert_babel`) and the frame-map logic of `get_frame_number_maps` from BehaveMAE
  (https://github.com/amathislab/BehaveMAE, commit `dd95e89`), copied verbatim. BehaveMAE is licensed under the Apache
  License 2.0; its license text is in `LICENSES/Apache-2.0-BehaveMAE.txt`.
- No other third-party code is included. The evaluators (BehaveMAE's `hierAS-eval`, the MABe 2022 public evaluator),
  TEACH's AMASS processing and split scripts, and the SMPL+H body model are used from their own repositories under their
  own licenses and are not redistributed here.

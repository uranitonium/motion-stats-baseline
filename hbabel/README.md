# Rebuilding hBABEL

hBABEL's benchmark files are not distributed, so they are rebuilt from public sources with the authors' own scripts.
Everything below runs those scripts unmodified, except TEACH's split script, whose three file paths point at your
folders. Downloads need accounts on the MPI sites and acceptance of their licenses; nothing from them is included here.

1. **Download.** From [AMASS](https://amass.is.tue.mpg.de), the SMPL+H G archives of the 17 subsets that BABEL's labels
   refer to (ACCAD, BMLmovi, BMLrub, CMU, DFaust, EKUT, EyesJapanDataset, HDM05, HumanEva, KIT, MoSh, PosePrior, SFU,
   SSM, TCDHands, TotalCapture, Transitions); TEACH's processed BABEL labels (`babel-teach`, see
   [TEACH](https://github.com/athn-nik/teach)); and the SMPL+H body model ([MANO](https://mano.is.tue.mpg.de)).
2. **Process AMASS** with TEACH's `scripts/process_amass.py --use-betas --gender male` (30 frames per second). We ran it in
   five groups of subsets in parallel and merged the outputs with `merge_amass.py`, which rebuilds exactly what a
   single run would have written:

       python merge_amass.py --groups processed_groups --teach teach/scripts/process_amass.py \
           --out processed_amass_smplh_male_30fps/amass.pth.tar

3. **Split** with TEACH's AMASS-BABEL split script (its three paths set to your folders). This writes
   `babel-smplh-30fps-male/{train,val}.pth.tar`.
4. **BehaveMAE's hBABEL scripts** at commit `dd95e89`, unmodified, from the data folder:
   `python -m datasets.hBABEL.prepare_hbabel_data` (it stops after train and val, since TEACH writes no test split) and
   `python datasets/hBABEL/prepare_hbabel_labels.py`. This gives 1,159 labeled validation sequences, the test set.
5. **Put the labels in the evaluator's layout.** The evaluator reads its own frame map, task list and split, and refuses a
   submission whose frame map differs; step 4 lists the same sequences in another order. `align_hbabel.py labels`
   rebuilds the label file in the evaluator's layout with BehaveMAE's labelling loop (checked first to reproduce
   BehaveMAE's own file on the original order):

       M=$HIER/hierAS-eval/eval/metadata/hBABEL
       python align_hbabel.py labels --babel_val babel-teach/val.json \
           --our_labels hBABEL/hbabel_val_test_actions_val_top_120_60_filtered.npy \
           --pose babel-smplh-30fps-male/val_proc_realigned_procrustes.pth.tar \
           --shipped_map $M/frame_number_map_val_filtered.npy --shipped_tasks $M/task_info_hBABEL_val_top_120_60.json \
           --out hBABEL_aligned/hbabel_val_test_actions_val_top_120_60_filtered.npy

   Submissions are moved into the same layout with `align_hbabel.py relayout` (see the main README). With these files the
   released hBehaveMAE reproduces its published score (16.95 against 16.98).

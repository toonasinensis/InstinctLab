# v010 perception source

Source: RobanS2-TPPO-Parkour `artifacts/runtime/depth_sm_history_v28/worktree/instinct_rl/instinct_rl/modules/`.
Corresponding experiment: `1103_2329__R8-LDSMN-v010__V28_Hist8_ZDetach_NoPush_4000Envs`.

Copied source, with repository formatter and local import changes only:

- `depth_encoder.py`: `DepthConv2dEncoder` from `depth_encoder.py` (tokenizer intentionally excluded).
- `depth_latent.py`: `DepthOnlyEncoder` from `supermimic/depth_latent.py`, and `mlp` from `supermimic/dynamics.py`.

PGMT changes live in `pgmt_*` subclasses. The geometric decoder uses the copied ELU MLP.
Masked moment/RMS-delta updates follow `LatentWorldModel.update_latent_stats`; constants are explicit `DepthCfg` fields.
Joint reconstruction/WM optimization and detached online targets follow
`HistoryInstinctMJFlowSuperMimicRunner` in `runners/depth_supermimic_runner.py`.
No visual EMA target, VAE sampling, or extra proprio encoder is added.

Original complete-file SHA-256 values (before extraction/formatting):

- `depth_encoder.py`: `c37b944a30eb5abe504aad6399f686bd54902dcdf45ebbc36a42572ac9fbd959`
- `supermimic/depth_latent.py`: `6bb4e05505476edc473fa3f5fe96162210fb6e4a3aacd237018739e9c7f8a3c5`
- `supermimic/dynamics.py`: `225c554ae1ab8bc5080152302fa54290b142a7262cf65fa6c26bd857711c6f9d`

# Rules for AI agents

This file applies to every AI coding agent working in this repository (Claude, Gemini,
Copilot, Cursor, Codex, and others). The owner reviews and merges. Follow these rules
unless the owner explicitly says otherwise in the current task.

## Workflow

1. **Never push to `main`.** Work on a branch and open a pull request. The owner merges.
2. **One topic per pull request.** A model swap, a README rewrite and a rename are three
   pull requests, not one.
3. **Run `pytest` before every push** (`pip install -r requirements-dev.txt`). CI must be
   green on the pull request before asking for review.
4. **Say what you could not check.** If a dataset, model host or test was unreachable,
   write that in the pull request instead of reporting the step as done.
5. Commit messages say what changed and why. Keep `Co-authored-by` lines for the agent.

## Models and accuracy numbers

The app ships two models: the CIFAR-10 classifier (`model_vit_cifar10.onnx`) and the
General mode CLIP encoder plus `data/general_label_embeddings.npz`.

6. **Every accuracy number comes from a script run, not from hand-written text.** CIFAR-10
   numbers come from `scripts/eval_cifar_test.py`, General mode numbers from
   `scripts/eval_general.py`. README numbers must match `results/*.json` exactly.
7. **A model or embedding change is only merged after the verify workflow ran on its
   branch.** `Verify model accuracy` (CIFAR-10) and `Verify General mode` run on branches
   other than `main` and commit their results to the branch. Do not write results by hand.
8. **The CIFAR-10 test set (10,000 images) is never used for training or for choosing a
   model, checkpoint, threshold or test-time policy.** Choose on a validation split taken
   from the training set, then measure on test once.
9. **Record provenance for third-party weights**: source URL with a pinned revision,
   SHA-256, license, base model and pretraining data, and which split the author evaluated
   on. Keep it in `results/provenance/`. Do not claim facts you have not checked: if it is
   unknown whether the test split was used, store `null`, not `false`.
10. **Check the license** of any downloaded weights before committing them. The repository
    is MIT; weights with a different or missing license must be stated in the README.
11. **Compare like with like.** When a pretrained model replaces one trained from scratch,
    say so next to the comparison.

## Reproducibility

12. **Every committed generated file must be reproducible from the repository.** Do not
    delete a script that builds a file still in use (for example
    `scripts/build_general_embeddings.py` builds the label embeddings). If you change the
    file, change or add the script that builds it.
13. Pin downloads to an exact revision or version and check the SHA-256 where possible.
14. Keep files under 100 MB (no Git LFS). Do not commit datasets, caches or `.part` files.

## Code

15. **Comments explain why, not what.** Do not shorten or delete comments that explain a
    threshold, a workaround or a pinned version (for example the `starlette` pin in
    `requirements.txt`).
16. User-facing text in the app is Indonesian. Keep it Indonesian unless the owner asks.
17. Keep dependency pins; upgrade one at a time in its own pull request with tests.
18. Never commit secrets. Use `.streamlit/secrets.toml`, which is ignored.

## Things only the owner decides

Renaming the project or repository, changing the license, deleting results or history,
replacing a model, and changing what the app shows to users. Propose these in a pull
request; do not do them as a side effect of another task.

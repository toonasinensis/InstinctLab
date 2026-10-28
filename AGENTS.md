# Agent Notes

## Project Shape

- Read `agents/CommonAgentInstructions.md` as the coding starting point.
- Read computation resources in `agents/computation/` before any verified compute or test workflow.
- If this is a MAC, run all tests and code on the external computation server, not on the local machine.

## External reference files

- Keep downloaded reference repositories, temporary dependency checkouts, and reference archives outside this repository.
- Use the sibling directory `<repository-name>-reference/` (currently `../InstinctLab-reference/`); do not recreate `.reference/` inside the checkout.
- Keep checkpoints, datasets, experiment reports, TensorBoard events, screenshots, and local runtime configuration out of Git. Store generated experiment files in the sibling `<repository-name>-local/` directory.

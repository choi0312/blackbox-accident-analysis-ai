# Third-party notices

Stage 1 uses the frozen Qwen3-VL-4B-Instruct-FP8 checkpoint under Apache-2.0.
The public FP8 weights are decoded to BF16 for local inference. Source and
revision details are preserved in `mllm_weights/provenance.json`.

Stage 2 uses YOLOPv2 for vehicle, drivable-area, and lane perception. The model
and adapted tensor decoding originate from
https://github.com/CAIC-AD/YOLOPv2 under the MIT license.

Stage 2 also uses two frozen SimpleTAD Video ViT checkpoints. Architecture
source: https://github.com/tue-mps/simple-tad at revision
`dce2c50e4ee126e38e962c1ca6177545628fdc68`; model source:
https://huggingface.co/tue-mps/simple-tad at revision
`a119df6fd334c8bdfd18ae6bed31e4b4b60cac74`. The included code and checkpoints
remain subject to CC-BY-NC-4.0 and their upstream notices.

Stage 3 runs as an independently callable GPL-3.0-only program and communicates
through a temporary CSV. Its FlexiNet source originates from
https://github.com/Geekgineer/FlexiNet at revision
`e25127f6912733213d46881d1139c9632e2f0114`. RAFT uses the torchvision
`C_T_SKHT_K_V2` checkpoint under the included BSD notices. Stage 3 also uses the
MIT-licensed YOLOPv2 checkpoint for lane segmentation.

All deployed checkpoints are local and frozen during inference. Original
authors retain their rights. Permission for public noncommercial competition
use does not relicense third-party works or grant commercial or exclusive
rights. The component license and notice files must remain with redistribution.

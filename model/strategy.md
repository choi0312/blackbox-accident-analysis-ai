# Final inference strategy

- Stage 1 samples 14 uniformly distributed frames and applies a fixed 0.46
  decision threshold.
- Stage 2 uses a dual-SimpleTAD contradiction guard around geometric contact
  estimation.
- Stage 3 combines frozen FlexiNet, RAFT, and YOLOPv2 representations with
  compact fixed heads and temporal decoding.
- Inference remains file-independent, offline, and free of test-time updates.


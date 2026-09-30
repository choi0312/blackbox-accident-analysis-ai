# Qwen3-VL MLLM positive-evidence guard

Qwen Team / Alibaba Cloud. Qwen3-VL-4B-Instruct-FP8, Apache-2.0.
https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8
https://github.com/QwenLM/Qwen3-VL
https://arxiv.org/abs/2511.21631

Publisher FP8 weights are unchanged on disk. Our modified loader decodes the 128x128 block scales to BF16 weights at startup and uses BF16 activations; it does not reproduce the native FP8 activation-quantized execution. The forensic classifier, file-local aggregation, and answer-order checks were added for this black-box video analysis pipeline. No external network is used in inference.

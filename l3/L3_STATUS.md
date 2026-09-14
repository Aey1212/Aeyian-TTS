# L3 status checkpoint

## Canonical base

- Canonical pronunciation/synthesis base: `fusion-v3-good` / F3.
- Canonical humanized result: `L3a-good` at commit `f8020ade3dd1cb1d7301098381fefb0a6fff8eec`.
- L3a uses OpenVoice V2 ToneColorConverter on F3 with the official OpenVoice example reference.
- User listening result: human-like speaker texture, no vinyl artifact, phonetic correctness retained at the M-branch level. Remaining issue is faint CRT/white-noise-like tingling and low perceived loudness.

## L3b HQ experiment

- Branch: `L3b-lossless-source`.
- Successful render head: `68a634fbd685252924fb32bd0eb40548455e7edb`.
- Exact lossless O15 and M1 inputs were verified before fusion.
- F3-HQ path: 88.2 kHz, mono, IEEE 64-bit float; fusion math retained in float64/complex128.
- Same OpenVoice V2 L3 conversion was then applied with no post-L3 correction.
- User listening result: L3a sounded more stable.
- Decision: L3b is rejected as the continuation base. Keep it only as a documented experiment.

## Current rule

All further L3 work continues from `L3a-good`, not L3b.

No post-L3 cleanup is allowed as the main strategy: no denoiser, EQ, phonetic re-harness, harmonic/formant polish, or similar correction after conversion. Humanization should be improved inside the voice-conversion stage itself while preserving F3 pronunciation.

## Next direction: L3c humanization

Planned first experiment on this branch:

1. Keep the L3a speaker/reference voice initially so only the extraction/conversion method changes.
2. Build a longer F3 calibration corpus that covers Commune Elven phonemes and important transitions instead of deriving the source embedding from one 2.2-second sentence.
3. Use segmented/VAD-cleaned reference speech and average several target-speaker embeddings instead of one embedding from the whole reference file.
4. Render a controlled OpenVoice `tau` sweep while keeping all other variables fixed.
5. Judge primarily by ear for human throat/voice texture, stability, and CRT/fizz character; reject any setting that harms F3 phonetic correctness.

L3a remains the safety anchor until a later version is clearly better by listening.

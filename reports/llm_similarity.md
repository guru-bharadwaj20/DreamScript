# Phase 12.3.5 - similarity of LoRA test outputs to their reference targets

`src.eval.similarity` over 162 test generations. CodeBLEU is the weighted mean of the components that are *defined* for a language - an undefined one is dropped and the rest renormalised, never scored as zero - so the per-language rows below are not comparable with each other, only arm against arm within a row.

| metric | LoRA | reference vs itself |
| :--- | :--- | :--- |
| codebleu | 0.887649 | 1.0 |
| ngram_match | 0.861899 | 1.0 |
| weighted_ngram_match | 0.856409 | 1.0 |
| syntax_match | 0.909684 | 1.0 |
| dataflow_match | 0.921655 | 1.0 |
| exact_match | 0.487654 | 1.0 |
| edit_similarity | 0.871703 | 1.0 |

## Per language

| language | n | codebleu | exact | edit_sim |
| :--- | :--- | :--- | :--- | :--- |
| python | 162 | 0.887649 | 0.487654 | 0.871703 |

## What the metric does to known perturbations of the references

The calibration that says how to read the table above.

| perturbation | codebleu | exact | edit_sim |
| :--- | :--- | :--- | :--- |
| identity | 1.0 | 1.0 | 1.0 |
| reformat | 1.0 | 1.0 | 1.0 |
| rename_locals | 0.5779 | 0.0 | 0.8276 |
| reverse_statements | 0.7973 | 0.0 | 0.3745 |
| drop_one_statement | 0.976 | 0.0 | 0.9743 |

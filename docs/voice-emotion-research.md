# Voice Emotion Research

## Decision

Use Piper as the robot's local speaking engine and `src/speech.py` as its explainable delivery planner. It changes pace and natural variation by short phrase according to the trusted robot reaction and punctuation. This is light enough to test on the laptop and later move to Raspberry Pi.

This is deliberately not voice cloning and does not infer emotion, identity, or a personal trait from a visitor's face.

## What Each Type of Data Would Train

| Goal | Appropriate data | Output | Status |
| --- | --- | --- | --- |
| Make the robot sound more lively now | No audio dataset required; trusted robot reaction and reply punctuation | `happy`, `proud`, `heart`, `confused`, and question delivery styles | implemented and audibly checked |
| Recognize broad vocal tone from a visitor later | Labeled emotional speech clips | A conservative optional signal for the face, never a diagnosis | not started; microphone calibration comes first |
| Create a new expressive robot voice | Consent-based recordings from one willing voice donor with text transcripts | A separate TTS model | out of scope for the Raspberry Pi prototype |

## Public Sources Considered

- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) has 7,442 English actor clips labeled with six emotions and intensity. It is a good candidate for a future speech-emotion-recognition experiment, but cloning its entire repository can require about 7.55 GB. We will select a small, balanced subset only after the microphone works.
- [EmoV-DB](https://openslr.org/115/) was created for emotional speech synthesis. It includes neutral, amused, sleepy, angry, and disgusted speech from four English speakers. Its archives are roughly 55--132 MB each, so it is not downloaded by default.
- [RAVDESS](https://smartlaboratory.org/resources/speech-song-database-ravdess/) is useful for a research-only emotion-recognition comparison, but its CC BY-NC-SA license means it must not become a commercial product asset.

## Why Not Fine-Tune XTTS Now

XTTS supports voice cloning and fine-tuning, but its official guide uses a GPU or hosted Colab workflow and notes a training run can take up to 40 minutes. It would create a heavier second voice system to maintain and test on the Pi. For this science-fair prototype, the immediate goal is reliable conversation, gesture reactions, and a transparent delivery policy rather than a cloned human voice.

## Dataset Guardrails

When the microphone is verified, download only the selected clips needed for a small, balanced experiment. Keep their source, license, label, and split in a manifest. Do not train on recordings without the speaker's permission, and do not use visitor audio to identify people or make claims about their mood.

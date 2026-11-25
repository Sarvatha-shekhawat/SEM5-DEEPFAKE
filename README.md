# DeepGuardAI

DeepGuardAI is a student project that screens videos for possible facial manipulation. It detects and crops faces from sampled frames, classifies each face with an EfficientNet-B0 model, and presents a video-level result through a Flask web application.

## Repository contents

- `Deepfake/` — Flask app, model inference, face extraction, templates, and trained weights.
- `Deepfake/dataset/` — labeled face crops used by the fine-tuning script.
- `archive/` — source samples arranged into real/MSVD and fake video-generation folders.
- `model.py`, `train.py`, `main.py` — earlier training implementation retained from the original project tree.
- `docs/` — IEEE-style project report.

The dataset folders contain 1,600 real and 1,088 fake face crops for training. The separate archive contains 1,600 real and 1,888 fake frames. These are image counts, not independent video counts; the repository does not document how the active training set was selected from the archive.

## Run the web application

Use Python 3.10 or later in a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install moviepy librosa
```

Set a persistent Flask session secret, then start the app:

```bash
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
cd Deepfake
python app.py
```

Open the local address printed by Flask. Keep `FLASK_SECRET_KEY` private and do not commit `.env` files or uploaded media.

## Fine-tuning

From `Deepfake/`, install the project dependencies and run:

```bash
python finetune_efficientnet.py
```

The script uses an image-level random validation split. For a meaningful performance estimate, split by source video and evaluate on videos withheld from training. No verified test-set results are included in this repository.

## Limitations

The model output is a screening aid, not proof that a video is authentic or manipulated. The app aggregates face-level scores using a fixed threshold; no calibration or independent benchmark results are provided. The audio analysis currently uses placeholder decision logic and is not a validated audio deepfake detector.

## Attribution

This project tree includes work derived from the existing [DeepGuardAI repository](https://github.com/PallaviGudupallavi/DeepGuardAI). The dataset attribution recorded in the project README points to the [Real/Fake Video Dataset on Kaggle](https://www.kaggle.com/datasets/mohammadsarfrazalam/realfake-video-dataset). Please retain applicable third-party dataset and code terms when redistributing.

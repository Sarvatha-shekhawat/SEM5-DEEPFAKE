# DeepGuardAI: A Face-Centric System for Deepfake Video Screening

**Sarvatha Shekhawat, Parth Gaur, and Preksha Gaur**  
Artificial Intelligence and Machine Learning Project · November 2026

**Abstract—** Deepfakes and other AI-generated facial videos can undermine trust in digital media. This paper presents DeepGuardAI, a prototype system that analyzes sampled video frames, detects and crops faces, and classifies the resulting face images as real or fake. The current implementation uses an ImageNet-pretrained EfficientNet-B0 with a binary classification head, together with MTCNN-based face extraction and a Flask interface for upload, inference, result history, and alerts. The repository contains 2,688 labeled face crops (1,600 real and 1,088 fake) in its training folders. Training configuration specifies an eight-epoch fine-tuning process with a 15% image-level validation split. However, the available evaluation routine reports only accuracy, and no reproducible metric output or held-out video-level evaluation is included. Accordingly, this paper describes the implemented pipeline without claiming a measured detection accuracy. The code also includes an audio-analysis path based on hand-crafted features and placeholder decision logic; it is not a validated audio deepfake detector. The system is therefore best understood as an educational prototype whose results require human review.

**Index Terms—** deepfake detection, EfficientNet-B0, face detection, video analysis, transfer learning, Flask.

## I. INTRODUCTION

Recent generative models make it easier to create convincing synthetic or manipulated facial media. A detector that helps a person review suspicious video can be useful in education, moderation, and media verification, but no classifier should be treated as proof of authenticity. Detection performance can fall when compression, lighting, identity, or manipulation methods differ from the training data. Prior work has emphasized the need for careful benchmarks and evaluation across manipulation types [1].

This project, named DeepGuardAI in its README, combines computer vision inference with a small web application. Given an uploaded video, the system samples frames, extracts a face where one is detected, resizes and normalizes each face crop, and applies a binary neural classifier. The interface reports an overall label and confidence-like score, can display frame-level scores and simple image measurements, and records analysis history. The project also includes a batch processing path and configurable alert hooks.

This report documents what is present in the supplied repository. It distinguishes implemented behavior from planned or simulated behavior, and it avoids reporting accuracy values that could not be verified from saved evaluation results.

## II. SYSTEM OVERVIEW

The application follows a sequential pipeline: (1) accept an image or video; (2) sample video frames; (3) locate and crop a face using MTCNN; (4) resize the crop to 224 × 224 pixels and apply ImageNet normalization; (5) infer a real/fake score with EfficientNet-B0; and (6) aggregate frame scores into a video-level result. The Flask app provides upload and batch routes, stores analysis metadata in SQLite, and can dispatch notifications when the predicted label is fake.

For a video with frame-level fake probabilities pᵢ, the inference code computes a weighted score p = 0.6 mean(pᵢ) + 0.4 max(pᵢ). The video is labeled fake when p ≥ 0.42; otherwise it is labeled real. The score is described in the interface as confidence, but it is a thresholded aggregate of model probabilities rather than a calibrated probability that the whole video is manipulated. If no face is found, the system returns an unable-to-detect result.

## III. METHODS

A. Face and frame processing: The production extraction routine uses a shared MTCNN detector. It estimates video duration from frame count and frame rate, aims to analyze roughly two frames per second, caps the target at 60 samples, and uses a minimum target of five frames for short clips where possible. A detected face is saved as an individual image. This sampling reduces computation, though it can miss brief manipulations between sampled frames.

B. Classifier: The model wraps torchvision EfficientNet-B0 initialized with ImageNet weights [2]. Its original classifier is replaced by dropout (p = 0.3) and a linear layer with two outputs. During inference, inputs are resized to 224 × 224 and normalized using the standard ImageNet channel means and standard deviations. The two logits are converted to softmax scores for real and fake classes.

C. Fine-tuning configuration: The included fine-tuning script collects images from dataset/real and dataset/fake, labels them 0 and 1, shuffles the combined image list using seed 42, and reserves 15% for validation. It specifies eight epochs, batch size 16, and initial learning rate 2 × 10⁻⁴. Random horizontal flips, color jitter, and rotations up to 10 degrees are applied to training images. The classifier head is trained alone for the first two epochs; all model layers are unfrozen beginning in epoch three, with a lower learning rate. The best validation-accuracy checkpoint is saved to model.pth.

The active training folders contain 1,600 real face images and 1,088 fake face images. The supplied ZIP also contains a separate archive of 1,600 real frames under an MSVD directory and 1,888 fake frames under directories named text2video-zero, Sora, and zeroscope. These are archived image counts, not independent video counts, and they do not match the active training-folder class totals. The repository does not document the transformation or selection that produced the active training set. Because the supplied split is performed at the image level after combining all active samples, neighboring crops from the same source video may occur in both training and validation sets. A video-disjoint split is needed for a more reliable estimate of generalization.

D. Video aggregation and explanation: The scoring module computes per-face scores and an aggregate as described above. For additional user context, it computes simple differences between consecutive grayscale crops, left-right brightness differences, and Laplacian variance as a texture/sharpness measure. These measurements support descriptive text in the interface; they are not learned forensic features and are not independently validated as evidence of manipulation.

## IV. IMPLEMENTATION

DeepGuardAI is implemented in Python with PyTorch/torchvision, OpenCV, facenet-pytorch, Pillow, and Flask. The application includes individual and batch upload flows, account-related routes, a SQLite analysis history, frame-level result details, and alert integration through Slack, email, or a custom webhook when credentials are configured. The code supports common video formats and image uploads. The database records filenames, labels, scores, timestamps, analyzed frame counts, and user IDs.

An audio route extracts a soundtrack and computes zero-crossing rate and spectral centroid. The current code then maps a simple arithmetic expression over these values to a score and threshold; its comments explicitly describe this as mock logic. It does not load or run a trained Wav2Vec2 detector. Audio results should therefore be regarded as illustrative output only, and are not included in this paper’s detection method or conclusions.

## V. EVALUATION AND DISCUSSION

The checked-in evaluation function runs inference over a dataset and prints accuracy. It does not produce precision, recall, F1-score, a confusion matrix, ROC-AUC, or calibration statistics. No evaluation log with measured values was present in the reviewed project files. Thus, this report does not state that the model achieves any particular accuracy, and the README’s general description of “high accuracy” cannot be substantiated from the available artifacts.

The current implementation nevertheless demonstrates an end-to-end prototype: a trained-weight file is present, the web app loads it for inference, and existing upload folders contain extracted face crops and processed video frames. These artifacts support the claim that the application has been exercised, but they do not establish benchmark performance. To assess effectiveness, evaluation should use a held-out set of complete videos, report class balance and video counts, compute precision, recall, F1-score, ROC-AUC, and confusion matrices, and test cross-dataset generalization. Threshold selection should be documented separately from final testing.

Several engineering details also affect reproducibility. The ZIP contains a nested DeepGuardAI.zip file in addition to the expanded project tree; this report is based on the expanded source and data tree. The fine-tuning script expects dataset paths relative to its current working directory, while the Flask app loads model.pth relative to its runtime directory. The repository contains older training and model entry points alongside the EfficientNet fine-tuning implementation, so the exact training path used to create the checked-in weights is not demonstrated by the code alone. The validation split should be grouped by source video to reduce leakage. These limitations do not prevent use as a demonstration, but they constrain scientific claims.

## VI. LIMITATIONS AND FUTURE WORK

The detector is face-centric and may fail on videos with no visible face, multiple people, unusual pose, occlusion, or low quality. Sampling up to 60 frames trades coverage for speed. A single face crop per sampled frame can discard contextual clues, and the aggregate threshold (0.42) is fixed in code rather than justified by a documented operating-point study. Training data provenance and diversity are described only generally in the README, which attributes a real/fake video dataset to Kaggle [3].

Future work should (1) create reproducible train, validation, and test partitions grouped by original video; (2) compare against a standard benchmark such as FaceForensics++ [1]; (3) report class-wise metrics and confidence intervals; (4) calibrate the decision threshold on validation data; (5) evaluate compression and unseen generators; (6) replace the mock audio branch with a separately trained and validated audio model; and (7) review deployment security, including secret-key management, upload retention, and notification configuration.

## VII. CONCLUSION

DeepGuardAI demonstrates a practical architecture for video face extraction and binary deepfake classification through a Flask application. Its principal technical components are MTCNN-based face crops, an EfficientNet-B0 classifier, weighted aggregation of frame scores, and a web interface that stores and presents analysis results. The project is a useful prototype and a starting point for systematic investigation. Since a verified test set and metric results are absent, its detection accuracy remains unknown. Treating its output as a review aid, and validating it with video-disjoint benchmarks, are essential next steps.

## REFERENCES

[2] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in Proc. 36th Int. Conf. Machine Learning, PMLR, vol. 97, 2019, pp. 6105–6114.

[3] M. Sarfraz Alam, “Real/Fake Video Dataset,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/mohammadsarfrazalam/realfake-video-dataset. Accessed: Nov. 26, 2026. Dataset attribution is taken from the project README.

[4] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks,” IEEE Signal Process. Lett., vol. 23, no. 10, pp. 1499–1503, Oct. 2016, doi: 10.1109/LSP.2016.2603342.


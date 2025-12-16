# DeepL / MINC-2500 Classification (Kaggle) — Notebook Guide

This repo contains a single end-to-end notebook (`classification_kaggle.ipynb`) that:
- Loads the **MINC-2500** materials dataset (Kaggle).
- Creates/uses a **train/val/test split**.
- Builds a TensorFlow input pipeline (**resize + normalize + augmentation**).
- Trains and evaluates multiple models:
  - **VGG19 from scratch** (TensorFlow/Keras)
  - **ResNet50 transfer learning** (TensorFlow/Keras)
  - **MobileNetV2 transfer learning** (TensorFlow/Keras)
  - **Inception V1 / GoogLeNet** (PyTorch via `torchvision`, wrapped for evaluation)
- Produces metrics, plots, Grad-CAM visualizations, and a model comparison table.

---

## Repo Contents

- **`classification_kaggle.ipynb`**: Main Kaggle-focused notebook.
- **`classification.ipynb`**: Alternate/local version (similar structure).
- **`requirements.txt`**: Python dependencies (mostly for local runs).
- **`LICENSE`**: License file.

---

## How to Run (Team Standard)

### Kaggle
- **Enable GPU**: Notebook → Settings → Accelerator → GPU.
- **Internet**:
  - If you want ImageNet pretrained weights for `ResNet50` / `MobileNetV2` / `GoogLeNet`, Internet must be **ON**.
  - If Internet is **OFF**, use `weights=None` (or `pretrained=False`) to avoid download errors.
- Run cells **top-to-bottom**. Many later cells depend on variables created earlier (paths, datasets, `CLASS_NAMES`, etc.).

### Local
- Ensure you have the dataset in the expected local folder and adjust `DATA_DIR` / `OUTPUT_DIR` in Cell 0.
- Install deps from `requirements.txt`.

---

## Notebook Walkthrough (What each section does)

### 1) Imports & Environment Setup (Cell 0)

The notebook imports:
- **Core**: `os`, `shutil`, `random`, `pathlib`, `glob`
- **Image IO**: `PIL.Image` (plus `ImageFile.LOAD_TRUNCATED_IMAGES=True` to tolerate some truncated images)
- **Math/Plotting**: `numpy`, `matplotlib`, `seaborn`
- **ML (TF/Keras)**: `tensorflow`, `tf.keras.layers`, Keras callbacks, `ResNet50`, `MobileNetV2`
- **Metrics**: `sklearn.metrics` (accuracy, PRF, report, confusion matrix, ROC/AUC)
- **CV**: `cv2` (used in Grad-CAM overlay)
- **Progress**: `tqdm`
- **PyTorch**: `torch`, `torchvision.models` (for GoogLeNet/Inception V1)

#### Critical point: GPU memory sharing (TF + PyTorch)
TensorFlow can aggressively allocate GPU memory. To reduce TF “hogging” VRAM (especially if you run PyTorch in the same session), the notebook sets:
- **`tf.config.experimental.set_memory_growth(gpu, True)`**

If this fails with an “already initialized” message, it’s usually safe; it just means TF already configured GPUs.

---

### 2) Dataset & Path Configuration (Cell 0)

Key variables:
- **`DATASET_NAME`**: Kaggle dataset slug. Example: `"minc2500/minc-2500"`.
- **`KAGGLE_INPUT`**: `/kaggle/input/<DATASET_NAME>`
- **`KAGGLE_WORKING`**: `/kaggle/working` (writable)
- **`IS_KAGGLE`**: `os.path.exists('/kaggle/input')`
- **`NEEDS_SPLIT`**: Whether the dataset needs a train/val/test split created.

Outputs / working directories:
- **`OUTPUT_DIR`**: location of prepared split (often `/kaggle/working/dataset`)
- **`MODELS_DIR`**: where models/checkpoints are saved
- **`REPORTS_DIR`**: where plots/reports/json are saved

---

### 3) Global Hyperparameters (Cell 0)

Key “team knobs”:
- **`IMG_SIZE = (224, 224)`**: Input image size for all CNNs.
- **`BATCH_SIZE = 64`**: Training batch size. Reduce if you hit GPU OOM.
- **`SEED = 42`**: Seed used to make operations reproducible *where it’s passed in*.
  - Used in dataset split creation (`random.seed(SEED)` and `random.shuffle`).
  - Used as `seed=` in `image_dataset_from_directory` for repeatable shuffling.
- **`MAX_EPOCHS = 4`**: Default training epoch count (Kaggle time-limited).

#### Important: What SEED does (and does not do)
- **Does**: Make Python `random`-based split creation repeatable and Keras dataset shuffling repeatable when `seed=SEED` is passed.
- **Does NOT** automatically seed everything:
  - If you need strict reproducibility, also set:
    - `np.random.seed(SEED)`
    - `tf.random.set_seed(SEED)`
    - and keep `shuffle` deterministic.

---

### 4) Data Integrity Helpers (Cell 1)

Functions:
- **`is_image_valid(path)`**:
  - Opens an image with PIL and calls `img.verify()`.
  - Returns `False` if PIL can’t parse/verify the file.
- **`remove_corrupted_images(data_dir)`**:
  - Walks through class folders and deletes broken images.

#### Critical point: “decode_image Input is empty”
If the dataset contains **0-byte files** or corrupted files, TF decoding can crash during training. A recommended team practice is to run cleanup on the prepared split directory before training.

---

### 5) Class Distribution & Split Stats (Cell 2)

Function:
- **`class_counts(data_dir)`**:
  - Returns a dict of `{class_name: image_count}` (filters by `.jpg/.jpeg/.png`).

Purpose:
- Verify the dataset is balanced.
- Verify train/val/test sizes after splitting.

#### Critical point: class mismatch across splits
The notebook output shows train/val/test may report different “classes found” counts (e.g., 14 vs 13). This often happens when:
- A split directory is incomplete/stale.
- Some class folders are missing in one split.
Team rule: **train/val/test must share the same set of class folders**.

---

### 6) Creating Train/Val/Test Split (Cells 3–4)

Function:
- **`create_split(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=SEED, force=False)`**

How it works:
- Sets `random.seed(seed)` and shuffles file paths per class.
- Copies images into:
  - `output_dir/train/<class>/...`
  - `output_dir/val/<class>/...`
  - `output_dir/test/<class>/...`
- If `force=False` and `output_dir/train` exists with files, it **skips** (reuses existing split).

Parameters:
- **`train_ratio`, `val_ratio`, `test_ratio`**: Must sum to 1.0.
- **`seed`**: Makes the split reproducible.
- **`force`**:
  - `False`: reuse existing split if present.
  - `True`: delete/rebuild for a clean split (recommended when class mismatch happens).

---

### 7) Input Pipeline (Preprocessing) (Cell 7)

Function:
- **`build_datasets(prepared_dir=OUTPUT_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, return_raw=False)`**

What it does:
- Builds `tf.data.Dataset`s using:
  - `tf.keras.preprocessing.image_dataset_from_directory(...)`
  - `label_mode='categorical'` → labels are one-hot vectors
  - `image_size=img_size` → resize
  - `batch_size=batch_size`
  - `shuffle=True` for training only
  - `interpolation='bilinear'`
- Adds normalization:
  - **`layers.Rescaling(1./255)`** → converts pixel range from `[0..255]` to `[0..1]`
- Performance tuning:
  - `.cache()` (stores dataset after preprocessing)
  - `.prefetch(tf.data.AUTOTUNE)` (overlaps CPU pipeline + GPU training)

Outputs:
- **`train_ds`**: normalized, batched, shuffled.
- **`val_ds`**, **`test_ds`**: normalized, batched, not shuffled.
- Optionally **`train_ds_raw`** (unnormalized) when `return_raw=True`.

Parameters:
- **`prepared_dir`**: directory containing `train/val/test`.
- **`img_size`**, **`batch_size`**, **`seed`**: pipeline configuration.
- **`return_raw`**: return raw dataset alongside normalized.

---

### 8) Data Augmentation (Cell 8)

Functions:
- **`get_augmentation_layer()`** returns a Keras `Sequential` augmentation model:
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.1)` (±10%)
  - `RandomZoom(0.1)`
  - `RandomTranslation(0.1, 0.1)`
  - `RandomContrast(0.1)`

Usage:
- Creates `train_ds_aug` by mapping the augmentation layer over `train_ds`.
- Augmentation is applied **only on training data**.

---

### 9) Class Names / Label Space (Cell 11)

The notebook sets:
- **`CLASS_NAMES`**: list of class folder names (sorted) from `OUTPUT_DIR/train`.
- **`NUM_CLASSES = len(CLASS_NAMES)`**

Critical: **Model output layer size must equal `NUM_CLASSES`**.

---

## Models (Architectures + How they operate)

### A) VGG19 From Scratch (Cell 16)

Builder:
- **`build_vgg19_scratch(input_shape=(224,224,3), num_classes=23, dropout_rate=0.5, l2_reg=1e-4)`**

Key architecture:
- 5 convolutional blocks with max-pooling:
  - **Block1**: Conv64 → Conv64 → MaxPool
  - **Block2**: Conv128 → Conv128 → MaxPool
  - **Block3**: Conv256 ×4 → MaxPool
  - **Block4**: Conv512 ×4 → MaxPool
  - **Block5**: Conv512 ×4 → MaxPool
- Classifier head:
  - Flatten → Dense(4096) → Dropout → Dense(4096) → Dropout → Dense(`num_classes`, softmax)

Important parameters:
- **`input_shape`**: fixed to `(224,224,3)` by default.
- **`num_classes`**: must equal `NUM_CLASSES`.
- **`dropout_rate`**: reduces overfitting by randomly dropping activations.
- **`l2_reg`**: L2 regularization strength on conv/dense kernels.

Initialization/regularization:
- **`HeNormal()`** initializer: good default for ReLU nets.
- **`regularizers.L2(l2_reg)`**: penalizes large weights.

Compile:
- Optimizer: **Adam(lr=1e-4)**
- Loss: **categorical_crossentropy** (because labels are one-hot)
- Metric: **accuracy**

---

### B) ResNet50 Transfer Learning (Cells 17 + 33)

Builder:
- **`build_resnet50_transfer(input_shape=(224,224,3), num_classes=23, include_top=False, weights='imagenet')`**

How it operates:
- Loads a pretrained ResNet50 backbone:
  - `include_top=False` removes ImageNet classification head.
  - `pooling='avg'` produces a global-average pooled feature vector.
- Freezes backbone first: **`base_model.trainable = False`**
- Adds a new classifier head:
  - Dense(512, relu) → Dropout(0.5) → Dense(num_classes, softmax)

Fine-tuning:
- **`unfreeze_resnet_for_finetuning(model, base_model, num_layers_to_unfreeze=40)`**
  - Unfreezes last N layers of the backbone, keeps earlier layers frozen.
  - Recompiles with **lower LR**: Adam(lr=1e-5).

Important parameters:
- **`weights`**:
  - `'imagenet'` gives best start but requires internet (if weights aren’t already cached).
  - `None` trains from random initialization.
- **`num_layers_to_unfreeze`**:
  - More layers unfreezed → more capacity but more risk of overfitting and slower training.

---

### C) MobileNetV2 Transfer Learning (Cell 18 + 35)

Builder:
- **`build_mobilenetv2_transfer(input_shape=(224,224,3), num_classes=23, include_top=False, weights='imagenet', alpha=1.0)`**

How it operates:
- Loads MobileNetV2 backbone (efficient CNN) with `include_top=False`, `pooling='avg'`.
- Freezes base model initially.
- Adds custom head:
  - Dense(512, relu) → Dropout(0.6) → Dense(num_classes, softmax)

Important parameters:
- **`alpha`**:
  - Width multiplier (smaller alpha → fewer channels → faster).
- **`weights`**: same internet/caching considerations as ResNet.
- Dropout is higher (0.6) to regularize more aggressively.

Compile:
- Adam(lr=1e-3) + categorical crossentropy + accuracy.

---

### D) Inception V1 / GoogLeNet (PyTorch) (Cell 19 + 37)

Wrapper:
- **`InceptionV1Wrapper(num_classes=23, pretrained=True)`**

What it does:
- Loads `torchvision.models.googlenet(...)`:
  - `weights=GoogLeNet_Weights.IMAGENET1K_V1` if pretrained else `None`
  - `aux_logits` is enabled when pretrained is used (per torchvision defaults).
- Replaces final classifier layer:
  - `self.model.fc = nn.Linear(num_features, num_classes)`
- Provides:
  - **`predict(images)`**: returns probability array (softmax)
  - **`save(path)`**, **`load(path)`**
  - **`get_trainable_params_count()`**

Training loop (custom PyTorch loop):
- Converts TF dataset to a streaming PyTorch iterable (to avoid materializing all data in RAM).
- Uses:
  - Optimizer: Adam(lr=1e-3)
  - Loss: CrossEntropyLoss
- Computes epoch loss by summing batch losses and dividing by counted batches (no `len()` on iterable loaders).

Critical points:
- Running TF + PyTorch together can cause GPU memory contention.
- If internet is disabled, pretrained weights loading may fail—use `pretrained=False`.

---

## Training Utilities (Cell 21)

Function:
- **`train_model(model, train_ds, val_ds, name, epochs=4, early_stopping_patience=5, reduce_lr_patience=3, initial_epoch=0, verbose=1, save_dir=None)`**

Callbacks:
- **ModelCheckpoint**:
  - Saves best model (by `val_loss`) to `MODELS_DIR/<name>.h5`
- **EarlyStopping**:
  - Stops training if `val_loss` doesn’t improve for N epochs.
  - `restore_best_weights=True` restores best weights.
- **ReduceLROnPlateau**:
  - Drops LR when `val_loss` plateaus to help convergence.

Key parameters:
- **`name`**: model checkpoint filename.
- **`epochs`**: max epochs.
- **`early_stopping_patience`**, **`reduce_lr_patience`**: convergence controls.
- **`save_dir`**: where checkpoints are written.

---

## Evaluation & Reporting (Cells 23+)

Core evaluation:
- **`get_predictions_and_labels(model, dataset, class_names=None)`**
  - Works for TF models and for the PyTorch wrapper (checks for `.predict` and `.device`).
- **`compute_classification_metrics(...)`**
  - Accuracy + per-class precision/recall/F1 + macro averages + sklearn classification report.
- **`compute_confusion_matrix(...)`**
  - Optionally normalized confusion matrix.
- **`compute_roc_curves(...)`**
  - One-vs-rest ROC curves per class + micro/macro ROC.
- **`evaluate_model(model, test_ds, model_name, class_names=None, save_dir=None)`**
  - Runs all the above and saves a JSON results file in `REPORTS_DIR`.

Visualization utilities:
- **`plot_training_curves(history, model_name, save_path=...)`**
- **`plot_confusion_matrix(cm, ...)`**
- **`plot_roc_curves(roc_data, ...)`**

Explainability:
- Grad-CAM helpers:
  - `get_last_conv_layer_name`, `make_gradcam_heatmap`, `overlay_heatmap`, `visualize_gradcam(...)`, `visualize_gradcam_samples(...)`

Model comparison:
- **`compare_models(results_list, ...)`**
  - Produces `model_comparison.csv` and `model_comparison.md`.

---

## Critical “Gotchas” (Team Checklist)

- **Class mismatch across splits**:
  - If `image_dataset_from_directory` reports different class counts for train/val/test, the split is inconsistent. Rebuild the split with `force=True` or delete `OUTPUT_DIR` and regenerate.
- **Corrupted/empty image files**:
  - Can crash TF decoding during training. Run cleanup before training if you see decode errors.
- **Kaggle Internet OFF**:
  - Pretrained weights downloads will fail (DNS/`gaierror`). Use `weights=None` (TF) and `pretrained=False` (PyTorch) or enable Kaggle Internet.
- **Reproducibility**:
  - `SEED` helps, but strict determinism requires also setting NumPy/TF seeds and avoiding nondeterministic ops.
- **GPU memory contention (TF + PyTorch)**:
  - Keep TF memory growth enabled, lower batch size, or run PyTorch section in a separate session if needed.

---

## Outputs (What gets saved where)

- **Models**: `MODELS_DIR` (Kaggle: `/kaggle/working/models`)
  - e.g. `vgg19_scratch.h5`, `resnet50_transfer.h5`, `mobilenetv2_transfer.h5`, `inception_v1.pth`
- **Reports**: `REPORTS_DIR` (Kaggle: `/kaggle/working/reports`)
  - Training curves, confusion matrices, ROC curves, Grad-CAM images
  - JSON metrics: `<model_name>_results.json`
  - Comparison: `model_comparison.csv`, `model_comparison.md`



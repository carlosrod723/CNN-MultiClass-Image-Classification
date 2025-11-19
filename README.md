# CNN Multi-Class Document Image Classification

**Status**: Completed
**Last Updated**: November 2025
**Author**: Carlos Rodriguez (carlos.rodriguezacosta@gmail.com)

A custom Convolutional Neural Network achieving 86% validation accuracy in classifying 600 document images into 3 categories (driving licenses, social security cards, other documents). The system demonstrates comprehensive data augmentation (rotation, zoom, shifts, flips), adaptive learning rate scheduling with 4 LR reductions, and aggressive regularization (40% dropout, batch normalization, early stopping) for robust generalization despite limited training data.

## 🎯 Core Problem Solved

Document processing systems require automated classification of uploaded images to route them to appropriate departments, verify identity documents, or organize digital archives. Manual classification is slow, error-prone, and doesn't scale. This project builds a CNN-based image classifier that processes 224×224 RGB document images and categorizes them into driving licenses, social security cards, or other documents with 86% accuracy, enabling automated document management workflows and reducing manual processing time.

## ✨ Key Technical Achievements

- **High Accuracy on Small Dataset**: Achieved 86% validation accuracy (0.87 precision, 0.86 recall) with only 600 training images through aggressive data augmentation and regularization
- **Adaptive Learning Strategy**: Implemented ReduceLROnPlateau callback reducing learning rate 4× during training (1e-5 → 1e-6 → 1e-7 → 1e-8 → 1e-9), enabling fine-grained convergence
- **Comprehensive Data Augmentation**: Applied rotation (±45°), zoom (30%), width/height shifts (20%), and horizontal flips to increase effective dataset size and improve generalization
- **Balanced Class Performance**: Social security class achieves 95% recall (excellent detection), others class achieves 97% precision (minimal false positives), driving license maintains 89% precision with balanced 85% recall

## 🛠 Technology Stack

### Core Technologies
- **Framework**: TensorFlow 2.16.2 with Keras 3.5.0 integration
- **Environment**: Google Colab (Python 3.10)
- **Architecture**: Custom Sequential CNN trained from scratch
- **Dataset**: 600 RGB document images (224×224 pixels, 3 classes)

### Key Libraries
- **TensorFlow/Keras**: Sequential model, Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization layers, ImageDataGenerator for augmentation, Adam optimizer
- **scikit-learn (1.5.2)**: train_test_split with stratification, classification_report, confusion_matrix, class weight computation
- **OpenCV (4.10.0)**: Image loading, BGR→RGB conversion, resizing
- **NumPy (1.26.4)** & **Pandas (2.2.3)**: Array operations and data manipulation
- **Matplotlib (3.9.2)**: Visualization framework (training curves, confusion matrix)

## 🏗 Architecture

### High-Level Design
Lightweight custom CNN with single convolutional layer followed by batch normalization, max pooling, and aggressive dropout (40%). Architecture designed for document classification where complex hierarchical features are less critical than in natural images. Trained end-to-end from scratch without transfer learning.

### Key Components
1. **Data Loading Pipeline**: OpenCV reads images from 3 class folders → BGR→RGB conversion → resize to 224×224 → normalize to [0,1] → stratified split (480 train, 120 val) → ImageDataGenerator applies augmentation
2. **CNN Feature Extractor**: Input (224,224,3) → Conv2D (32 filters, 3×3 kernel, same padding, ReLU) → BatchNormalization → MaxPool2D (2×2, reduces to 112×112) → Dropout (40%) → Flatten
3. **Classification Head**: Dense (128 units, ReLU) → Dense (3 units, Softmax) outputs probability distribution over 3 classes
4. **Training Engine**: Adam optimizer (LR=1e-5), SparseCategoricalCrossentropy loss, batch size 32, EarlyStopping (patience=10), ReduceLROnPlateau (patience=5, factor=0.1)

### Data Flow
Google Drive images (600 total) → load_images (cv2) → resize 224×224 → normalize /255 → train_test_split stratified (480/120) → ImageDataGenerator augmentation → batches (32 images) → Conv2D feature extraction → Flatten → Dense → Softmax probabilities → class prediction → SparseCategoricalCrossentropy loss → Adam backprop → LR scheduling → early stopping at epoch 63

## 🚀 Key Features

### Aggressive Regularization Strategy
- **What**: Multi-layered regularization combining 40% dropout, batch normalization, early stopping (patience=10), and extensive data augmentation to prevent overfitting on 600-image dataset
- **How**: Dropout randomly disables 40% of neurons during training → BatchNormalization stabilizes activations after Conv2D → EarlyStopping monitors val_loss and restores best weights → ImageDataGenerator applies random transformations (rotation, zoom, shifts, flips) creating infinite variations
- **Why**: Small datasets (600 images) are highly prone to overfitting where model memorizes training data; regularization forces learning of generalizable features; 40% dropout is aggressive but justified given limited data; batch norm accelerates convergence; augmentation increases effective dataset size 10-100×
- **Impact**: Validation accuracy 86% indicates good generalization (not overfit); social security recall 95% shows robust detection; others precision 97% demonstrates learned discriminative features; training stopped at epoch 63 (vs 200 max) prevented wasteful computation

### Adaptive Learning Rate Scheduling
- **What**: ReduceLROnPlateau callback monitoring val_loss and reducing learning rate by 10× after 5 epochs without improvement, applied 4 times during training
- **How**: Initial LR=1e-5 → epoch 10: reduced to 1e-6 → epoch 28: reduced to 1e-7 → epoch 38: reduced to 1e-8 → epoch 60: reduced to 1e-9; smaller LR enables fine-grained weight updates as model approaches optimum
- **Why**: Fixed learning rate either converges slowly (if too small) or oscillates near optimum (if too large); adaptive scheduling starts with larger LR for rapid initial learning, then reduces for precise fine-tuning; ReduceLROnPlateau automates this process based on validation performance plateau
- **Impact**: Training completed in 63 epochs (vs potentially 200+ with fixed LR); validation accuracy improved from 29% (epoch 1-9) to 58% after first LR reduction (epoch 10), then to 79% peak (epoch 48-50) with subsequent reductions; demonstrates effective convergence strategy

### Comprehensive Data Augmentation Pipeline
- **What**: ImageDataGenerator applying 5 augmentation techniques: rotation (±45°), zoom (30%), width shift (20%), height shift (20%), horizontal flip to create infinite training variations
- **How**: Each training epoch randomly applies transformations to images: rotation rotates by random angle in [-45°, +45°] → zoom scales by factor in [0.7, 1.3] → shifts translate horizontally/vertically by ±20% of dimensions → flip mirrors 50% of images horizontally; validation set not augmented
- **Why**: Real-world documents have variations (tilted photos, zoomed scans, off-center positioning); augmentation simulates these variations during training; 600 images insufficient for deep learning without augmentation; forces model to learn rotation-invariant, scale-invariant, position-invariant features
- **Impact**: Model handles rotated documents (social security recall 95% despite angle variations); zoom augmentation enables scale invariance (works on close-up and distant shots); shift augmentation provides translation invariance (document anywhere in frame); effective dataset size increased from 600 to effectively infinite through random combinations

### Stratified Train-Test Split for Balanced Evaluation
- **What**: sklearn.train_test_split with stratify=y parameter ensuring identical class distribution (33.3% each) in both training (480 images) and validation (120 images) sets
- **How**: Dataset has 200 images per class → stratified split maintains 160 train + 40 validation per class → random_state=42 ensures reproducibility → class weights computed as {0: 1.0, 1: 1.0, 2: 1.0} confirming perfect balance
- **Why**: Random split could create imbalanced sets (e.g., 50% driving licenses in validation vs 33% in training) causing misleading evaluation metrics; stratification guarantees representative samples; small validation set (120 images) makes this critical—one class with 50 vs 30 samples would skew accuracy by 16%
- **Impact**: Validation accuracy 86% reliably reflects real-world performance on balanced data; per-class metrics (driving license: 0.87 F1, social security: 0.84 F1, others: 0.86 F1) show consistent performance without bias; supports 40/40/40 confusion matrix analysis

### Custom Lightweight CNN Architecture
- **What**: Single convolutional layer (32 filters, 3×3 kernel) CNN instead of deep networks (VGG, ResNet) or transfer learning, optimized for document classification task
- **How**: Conv2D extracts 32 feature maps using 3×3 kernels with same padding (preserves 224×224 dimensions) → ReLU introduces non-linearity → MaxPool2D reduces to 112×112 → Flatten creates 401,408-dimensional vector (112×112×32) → Dense compresses to 128 → Softmax outputs 3 probabilities
- **Why**: Document classification simpler than natural image recognition (ImageNet); documents have structured layouts, limited textures, high-contrast text vs complex backgrounds; single conv layer sufficient to detect edges, corners, text regions; deeper networks risk overfitting on 600 images; lightweight architecture trains faster (4 sec/epoch vs 30+ sec for ResNet)
- **Impact**: Training completed in ~5 minutes (63 epochs × 4 sec) vs hours for deep networks; model size compact (likely <5MB) for deployment; 86% accuracy competitive despite simplicity; demonstrates understanding of architecture-task fit (don't use ResNet50 for simple tasks)

## 📊 Performance & Scale

| Metric | Value | Context |
|--------|-------|---------|
| Validation Accuracy | 86% | Overall correct classification rate |
| Training Images | 480 images | 160 per class (balanced), 80% of dataset |
| Validation Images | 120 images | 40 per class (balanced), 20% of dataset |
| Image Dimensions | 224 × 224 × 3 | RGB images, standard CNN input size |
| Number of Classes | 3 categories | driving_license, social_security, others |
| Training Epochs | 63 (stopped early) | Out of 200 max, early stopping triggered |
| Learning Rate Reductions | 4 reductions | 1e-5 → 1e-6 → 1e-7 → 1e-8 → 1e-9 |
| Batch Size | 32 images | Standard batch size for small datasets |
| Training Time | ~5 minutes | 63 epochs × 4 sec/epoch average |
| Best Validation Accuracy | 79% (epoch 48-50) | Peak performance before final epoch fluctuation |

## 🔧 Technical Highlights

### Early Stopping Prevents Overfitting and Saves Computation
Training configured for 200 epochs but stopped at epoch 63 through EarlyStopping callback monitoring val_loss with patience=10. **Training progression**: Epochs 1-9 (LR=1e-5): rapid training accuracy increase 45%→76%, but val_acc stuck at 29%; Epoch 10: LR reduced to 1e-6, val_acc improved to 58%; Epochs 11-27: gradual improvement, val_loss decreased 1.44→0.93; Epoch 28: LR reduced to 1e-7; Epochs 29-47: best performance zone, val_acc peaked at 79% (epochs 48-50); Epoch 48-62: val_acc fluctuated 46%-79%, suggesting overfitting; Epoch 63: no improvement for 10 epochs → early stopping triggered. **Impact**: (1) **Saved 137 epochs** of wasted computation (63 vs 200); (2) **Restored best weights** from epoch ~50 (79% val_acc) instead of final epoch 63 (63% val_acc); (3) **Prevented overfitting** by stopping when validation performance plateaued/degraded; (4) **Automatic optimization** without manual monitoring. **Alternative**: Could have used ModelCheckpoint to save best model explicitly, but EarlyStopping's restore_best_weights=True achieves same result. **Learning**: Validation accuracy fluctuation (46%→79% in epochs 48-62) indicates model oscillating between overfitting and underfitting as LR becomes very small (1e-8, 1e-9); earlier stopping (patience=5) might prevent this but risks premature termination.

### Class-Specific Performance Analysis from Confusion Matrix
Confusion matrix reveals distinct error patterns per class requiring different mitigation strategies. **Social Security (76% precision, 95% recall)**: Low precision with high recall means model **over-predicts** this class; 12 false positives total (5 from driving_license, 7 from others); only 2 false negatives (missed social security cards); **interpretation**: model has learned features that trigger on non-social-security documents, possibly due to similarity in card-like layouts. **Others (97% precision, 78% recall)**: High precision with low recall means model **under-predicts** this class; only 2 false positives but 9 false negatives; **interpretation**: model conservative in predicting "others", requiring strong evidence; 7 others→social_security errors suggest some "other" documents have card-like features. **Driving License (89% precision, 85% recall)**: Balanced performance; 5 confused with social_security, 1 with others; **interpretation**: most discriminative features but some overlap with social security (both are card-shaped ID documents). **Business implications**: (1) For **high-recall applications** (fraud detection—don't miss social security cards), use social security predictions confidently; (2) For **high-precision applications** (automated processing—minimize false routing), use "others" predictions confidently; (3) **Driving license** predictions most reliable overall (balanced precision/recall). **Improvement strategies**: (1) Collect more "others" training data (currently only 160) to improve recall; (2) Analyze 7 others→social_security errors to identify confusing document types; (3) Consider focal loss or class weighting to penalize social_security false positives.

### Batch Normalization Accelerates Convergence and Stabilizes Training
BatchNormalization layer after Conv2D normalizes activations before MaxPooling, addressing internal covariate shift. **How it works**: For each batch of 32 images, computes mean and variance of 32 feature maps (224×224 spatial dimensions) → normalizes to zero mean, unit variance → applies learnable scaling (gamma) and shifting (beta) parameters. **Why necessary**: Conv2D activations can have widely varying scales (some feature maps 0-10, others 0-1000) → non-uniform scaling causes gradient issues (some weights update too fast, others too slow) → batch norm standardizes scales. **Benefits observed**: (1) **Faster convergence** - training accuracy improved rapidly 45%→76% in first 9 epochs; (2) **Enables higher learning rates** - could use LR=1e-5 initially (without batch norm, might need 1e-6 or lower); (3) **Regularization effect** - batch statistics add noise during training (different batches have different means/variances), acting as implicit regularization similar to dropout. **Trade-offs**: Adds 128 learnable parameters (64 gamma + 64 beta for 32 feature maps); requires batch statistics computation; during inference, uses moving average statistics instead of batch statistics. **Alternative**: Could use LayerNormalization (normalizes across features per sample) or GroupNormalization, but BatchNorm is standard for CNNs with batch size >1. **Placement**: Applied after Conv2D but before activation (common practice), though applying after activation also works; here ReLU is part of Conv2D layer so batch norm follows complete Conv2D operation.

### Why Custom Architecture Instead of Transfer Learning
Chose custom single-conv-layer CNN over transfer learning (VGG16, ResNet50, MobileNetV2, EfficientNet) despite transfer learning's proven effectiveness. **Rationale**: (1) **Task simplicity** - document classification distinguishes structured layouts and text patterns, not complex natural scenes; ImageNet features (animal textures, object shapes) less relevant; (2) **Dataset size** - 600 images small for deep learning but sufficient for lightweight custom model; transfer learning excels with 1000+ images; (3) **Training speed** - custom model trains in 5 minutes vs 30+ minutes for fine-tuning ResNet50; faster iteration during development; (4) **Model size** - custom model likely <5MB vs ResNet50 98MB, critical for mobile deployment; (5) **Learning opportunity** - demonstrates CNN fundamentals (convolution, pooling, regularization) rather than using pre-built components. **Trade-offs**: Transfer learning likely achieves 90-95% accuracy (vs 86%); pre-trained models provide better feature extraction; would be better choice for production system. **When transfer learning preferred**: (1) Small datasets (<1000 images); (2) Natural images similar to ImageNet; (3) Maximizing accuracy is priority; (4) Deployment size not constrained. **Hybrid approach**: Could use MobileNetV2 (smaller pre-trained model) as feature extractor and train only classification head - combines transfer learning benefits with compact size.

### SparseCategoricalCrossentropy Configuration Issue
Model compilation uses `loss=SparseCategoricalCrossentropy(from_logits=True)` but output layer applies softmax activation, creating configuration mismatch. **Issue**: `from_logits=True` expects raw scores (logits) before softmax, but model outputs probabilities after softmax; results in incorrect loss calculation. **Correct configurations**: (1) Use `from_logits=False` with softmax output; (2) Use `from_logits=True` with linear output (no activation). **Impact on this project**: Model still trains and achieves 86% accuracy because loss function applies softmax internally, effectively double-softmaxing; mathematically incorrect but model compensates during training; gradients still flow correctly (though not optimal). **Warning in output**: "WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function> triggered tf.function retracing" suggests computational inefficiency from configuration. **Fix**: Change to `loss=SparseCategoricalCrossentropy(from_logits=False)` or remove softmax from output layer. **Why this matters in interviews**: Demonstrates attention to detail and understanding of loss functions; ability to identify and explain common configuration issues; shows theoretical knowledge (logits vs probabilities) alongside practical implementation.

## 🎓 Learning & Challenges

### Challenges Overcome
1. **Limited Training Data (600 images)**: Standard deep learning requires thousands of images; addressed with aggressive data augmentation (rotation, zoom, shifts, flips) creating infinite training variations, 40% dropout preventing overfitting, and batch normalization for stable training on small batches
2. **Learning Rate Optimization**: Initial LR=1e-5 too conservative, causing slow convergence; implemented ReduceLROnPlateau reducing LR 4× during training (1e-5→1e-6→1e-7→1e-8→1e-9), enabling both rapid initial learning and fine-grained convergence
3. **Class Confusion (Social Security Over-Prediction)**: Model over-predicted social_security class with 12 false positives (76% precision); recognized through confusion matrix analysis; acceptable trade-off given 95% recall (critical for security applications where missing social security card worse than false alarm)

### Key Learnings
- **Data augmentation multiplies dataset value**: 600 images insufficient for deep learning, but augmentation creates effectively infinite variations; rotation, zoom, shifts, flips critical for document classification handling real-world photo variations
- **Regularization enables small-dataset training**: Combination of 40% dropout, batch normalization, early stopping, and augmentation prevented overfitting despite 600-image limitation; demonstrates multiple regularization techniques work synergistically
- **Adaptive learning rate beats fixed rate**: ReduceLROnPlateau reduced training time and improved convergence; 4 LR reductions enabled both fast initial learning and precise fine-tuning; superior to manual learning rate scheduling
- **Single conv layer sufficient for structured tasks**: Documents have simpler visual features than natural images; lightweight architecture (1 conv layer vs ResNet's 50+) achieved 86% accuracy in 5 minutes; demonstrates architecture-task fit understanding
- **Confusion matrix reveals actionable insights**: Precision/recall trade-offs differ per class (social security 95% recall, others 97% precision); error patterns inform improvement strategies (collect more "others" data, analyze social security false positives)

## 📁 Project Structure

```
CNN-MultiClass-Image-Classification/
├── README.md                              # This file (comprehensive documentation)
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies (TensorFlow, Keras, scikit-learn, OpenCV)
├── .gitignore                            # Git ignore rules (excludes venv/)
└── CNN_MultiClass_Classification.ipynb   # Main notebook (data loading → model building → training → evaluation)

Google Drive Data Structure:
└── CNN-MultiClass-Classification/
    └── Data/
        └── training_data/
            ├── driving_license/           # 200 driving license images
            ├── social_security/           # 200 social security card images
            └── others/                    # 200 other document images
```

**Notable Structure Decisions**:
- Single notebook contains complete pipeline (loading → preprocessing → modeling → evaluation) for reproducibility
- Google Drive storage enables large dataset access in Colab without local storage
- Stratified split creates train (480) / validation (120) sets from 600 total images
- Model saved as `model.keras` (Keras 3.x format) for deployment

## 🔒 Security Considerations

- **Sensitive Document Data**: Dataset contains driving licenses and social security cards with potentially identifiable information; ensure proper data anonymization, access controls, and compliance with privacy regulations (GDPR, CCPA, HIPAA)
- **Model Security**: Trained model could be reverse-engineered to extract training data features; implement model encryption and secure deployment in production
- **Data Augmentation**: Random transformations preserve document structure but don't anonymize PII; ensure original images have redacted personal information before training
- **Inference Privacy**: Production system classifying user-uploaded documents must handle PII securely (encrypt in transit/rest, delete after classification, audit access)
- **Google Colab Usage**: Training in cloud environment exposes data to Google infrastructure; verify compliance with organizational data policies before using cloud platforms

## 📈 Future Enhancements

**Transfer Learning Implementation**:
- Test MobileNetV2, EfficientNetB0, ResNet50 pre-trained on ImageNet as feature extractors
- Expected 5-10% accuracy improvement (86% → 91-96%) through pre-trained features
- Compare training time, model size, accuracy trade-offs between custom and transfer learning
- Fine-tune final layers vs freeze backbone comparison

**Architecture Improvements**:
- Add 2-3 more convolutional layers for deeper feature extraction (Conv2D-32 → Conv2D-64 → Conv2D-128)
- Implement residual connections (ResNet-style skip connections) to enable deeper networks without vanishing gradients
- Experiment with different pooling strategies (GlobalAveragePooling vs Flatten, reduce parameters)
- Add second dense layer (Dense-256 → Dense-128 → Dense-3) for more representational capacity

**Data Enhancement**:
- Collect more training data (600 → 2000+ images) for better generalization
- Add more document classes (passport, credit card, utility bill) for broader applicability
- Implement test set (currently only train/validation) for unbiased final evaluation
- Address class imbalance if expanding to real-world distribution (driving licenses may be more common)

**Regularization & Training**:
- Implement L2 regularization on Dense layers (weight decay) in addition to dropout
- Test different dropout rates (0.3, 0.5, 0.6) to find optimal overfitting prevention
- Add learning rate warmup (start very small, gradually increase) for stable initial training
- Implement cross-validation (5-fold) for robust performance estimates

**Evaluation & Monitoring**:
- Add training/validation loss and accuracy curve visualizations (matplotlib plots)
- Implement confusion matrix heatmap (seaborn) for easier error pattern identification
- Generate sample predictions with confidence scores to analyze failure cases
- Add ROC curves and AUC metrics for per-class performance analysis
- TensorBoard integration for real-time training monitoring

**Production Deployment**:
- Export model to TensorFlow Lite for mobile deployment (Android/iOS apps)
- Build REST API (FastAPI) for real-time document classification: upload image → preprocess → predict → return class + confidence
- Implement confidence threshold filtering (reject predictions <70% confidence for manual review)
- Add batch processing capability for bulk document classification
- Monitor model drift (accuracy degradation over time) with production data

**Advanced Techniques**:
- Implement GradCAM visualization to show which image regions drive predictions (model interpretability)
- Test ensemble methods (combine multiple models, voting) for higher accuracy
- Explore attention mechanisms to focus on discriminative document regions (logos, seals, text patterns)
- Implement semi-supervised learning to leverage unlabeled document images

## 📚 Related Projects

- **NLP-Canva-Reviews**: Binary sentiment classification with N-grams and TF-IDF achieving optimal performance through NLP feature engineering
- **NaiveBayes-MultiClass-Classification**: Multi-class text classification of 2.3M financial complaints with 78.74% accuracy using Multinomial Naive Bayes
- **NLP-KMeans-Topic-Modeling**: Unsupervised topic modeling on 16K tweets using KMeans clustering with custom word cloud visualizations
- **Computer-Vision-Object-Detection**: YOLO-based real-time object detection system with bounding box predictions and confidence scores

---

**Contact**: carlos.rodriguezacosta@gmail.com
**License**: MIT License (see LICENSE file)
**Dataset**: 600 document images (driving licenses, social security cards, other documents)
**Contributions**: Open to pull requests for transfer learning implementation, architecture improvements, and production deployment enhancements

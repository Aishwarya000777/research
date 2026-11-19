Linear Classifier: An Often-Forgotten Baseline for Text Classification
A reproduction-oriented implementation of the paper "Linear
Classifier: An Often-Forgotten Baseline for Text Classification" (ACL
2023), evaluated on the SCOTUS subset of the LexGLUE benchmark.

This project implements and compares three linear classification
approaches:
1. One-vs-Rest (OVR)
2. Thresholding
3. Cost-Sensitive Learning
It contains an initial scikit-learn implementation and a second
implementation using the authors' intended LibMultiLabel training
routines. A Streamlit application lets you enter a legal document and
inspect predictions from all six saved models.
Reproduction note: This is a scoped reproduction, not a claim of
bit-for-bit reproduction. The official-method results are close to the
paper's reported SCOTUS results. Differences in library versions,
preprocessing/vectorization details, and environment-level
compatibility patches can affect exact scores. The comparison below
reports the actual results produced by this project.

1. Project motivation
Text classification assigns a document to one or more predefined
categories. Legal judgments are long, domain-specific documents, and
class distributions can be highly imbalanced.

The paper investigates linear classifiers as strong and often overlooked
baselines for text classification. The goal of this project is to
reproduce the paper's SCOTUS experiment and understand how three
approaches behave:
- OVR: trains one binary classifier per class and selects the
  class with the strongest score.
- Thresholding: learns class-specific decision thresholds rather
  than relying only on the default zero threshold.
- Cost-sensitive learning: incorporates class imbalance/cost
  considerations during training so minority classes are not treated
  identically to majority classes.
The project also provides an interactive demo to classify a pasted legal
judgment.
2. Dataset
We use the scotus configuration of the Hugging Face dataset:
from datasets import load_dataset

dataset = load_dataset("coastalcph/lex_glue", "scotus")
Dataset fields used:
- text: the legal judgment text.
- label: the target class, represented by integer IDs from 0 to
  12.
The dataset contains 13 classes. The dataset split sizes observed in
this run were:

  Split          Documents
  Train              5,000
Validation         1,400
Test               1,400

Following the authors' stated protocol for linear methods, the training
and validation splits are combined:
- Final training set: 6,400 documents
- Final test set: 1,400 documents
The test set is kept separate from training and is used for final
evaluation.

The dataset's class names are shown as 1 through 13 in its metadata.
In the Streamlit interface, predictions are displayed as Class 1
through Class 13. The displayed class number is the dataset label ID
plus one.

The SCOTUS data is downloaded automatically by the Hugging Face
datasets library when the experiment is run for the first time.
Internet access is therefore required for the initial download.
3. Methods and implementation
3.1 Text preprocessing
The project keeps two preprocessing paths:
- preprocess_text: used by the original scikit-learn pipeline. It
  lowercases text, normalizes whitespace, and removes punctuation.
- preprocess_official_text: used by the LibMultiLabel experiment and
  Streamlit's official-model path. It normalizes whitespace without
  explicitly removing punctuation or lowercasing in this function. The
  TF-IDF vectorizer performs its own default lowercasing.
Keeping the two paths separate allows the original implementation and
the official-method implementation to use their respective preprocessing
pipelines.
3.2 TF-IDF feature extraction
Legal documents are converted into numerical feature vectors using
TfidfVectorizer.

The configured vectorizer uses unigram features and leaves stop words
enabled (that is, it does not remove stop words). It is fitted on the
combined training text and then used to transform the test text. The
same fitted vectorizer is saved with each implementation's model
artifacts.

The official experiment in the final run produced:
- Training matrix: 6400 × 126405
- Test matrix: 1400 × 126405
3.3 Classification approaches
  Approach                            Description
  OVR                                 One binary classifier per class;
choose the class with the highest
decision score.

  Thresholding                        Learns a separate threshold for
each class and applies those
thresholds to class scores.
Cost-sensitive                      Uses a cost-sensitive training
procedure to account for class
imbalance.
Two implementations are retained:
- Original implementation: scikit-learn OneVsRestClassifier with
  LinearSVC, plus the project's own weighted and out-of-fold
  thresholding variants. These are useful comparison baselines, but
  the custom thresholding implementation is an approximation and
  should not be described as the paper's exact thresholding algorithm.
- Official-method implementation: LibMultiLabel's train_1vsrest,
  train_thresholding, and train_cost_sensitive routines.
The official LibMultiLabel models use the SCOTUS solver options used in
the authors' configuration:
-s 2 -B 1 -e 0.0001 -q
3.4 Evaluation
The project reports:
- Accuracy
- Micro-F1
- Macro-F1
- Weighted-F1
- Macro precision
- Macro recall
Confusion matrices are also saved for visual inspection.

For this single-label, single-prediction task, accuracy and micro-F1
are equal. Macro-F1 is especially useful because it gives each class
equal weight, including classes with relatively few examples.
4. Results: paper vs. this implementation
The following table compares the paper's reported SCOTUS results with
the latest official LibMultiLabel run in this repository.
  Method           Paper Micro-F1   Our Micro-F1 Paper Macro-F1   Our Macro-F1
  OVR                        78.1          78.07           68.9          68.89

  Thresholding               78.9          78.07           71.5          69.77
Cost-sensitive             78.3          78.07           71.5          71.27
Scores are percentages. The project metrics are stored as fractions in
the CSV file (for example, 0.7807 means 78.07%).
Interpretation
- OVR: the reproduced result is extremely close to the paper for
  both Micro-F1 and Macro-F1.
- Thresholding: the Macro-F1 is lower than the paper's result in
  this run. The Micro-F1 is also lower.
- Cost-sensitive: the Macro-F1 is close to the paper's reported
  result, while Micro-F1 is lower.
These results indicate that the implementation captures the main
behavior of the approaches and obtains results close to the paper, but
does not exactly reproduce every reported score. Differences can
arise from preprocessing, vectorizer/library behavior, package versions,
solver/runtime details, and implementation/environment differences.

Do not describe the results as an exact match. Use the saved CSV as the
source of truth for the run in your local copy.
Where the results are saved
The latest official experiment outputs are in:
results/official/
├── official_metrics.csv
├── official_baseline_confusion_matrix.png
├── official_thresholded_confusion_matrix.png
├── official_cost_sensitive_confusion_matrix.png
└── official_model_artifacts.pkl
official_metrics.csv contains the measured metrics. The
confusion-matrix PNG files show per-class prediction behavior.
official_model_artifacts.pkl stores the official vectorizer and
trained models for the demo.

The original implementation's outputs are stored separately under
results/ (including test_metrics.csv, confusion_matrix.png,
test_confusion_matrix.png, and model_artifacts.pkl, depending on the
experiment run).
5. Repository structure
linear-classifier-reproduction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── (dataset/cache or locally stored data, if used)
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   ├── thresholding.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── official_models.py
│   └── official_experiments.py
│
└── results/
    ├── metrics.csv
    ├── test_metrics.csv
    ├── confusion_matrix.png
    ├── test_confusion_matrix.png
    ├── model_artifacts.pkl
    │
    └── official/
        ├── official_metrics.csv
        ├── official_baseline_confusion_matrix.png
        ├── official_thresholded_confusion_matrix.png
        ├── official_cost_sensitive_confusion_matrix.png
        └── official_model_artifacts.pkl
What each file does
  File                                              Purpose
  app.py                                          Streamlit interface. Loads both
artifact bundles, lets the user
choose one of six models, accepts a
legal document, predicts a class,
and displays decision scores and
processing time.

  requirements.txt                                Python dependencies required by the
project.

  notebooks/exploration.ipynb                     Initial dataset exploration and
inspection.

  src/dataset.py                                  Downloads/loads SCOTUS and displays
dataset information, split sizes,
label distribution, and sample
records.

  src/preprocessing.py                            Contains the original and
official-path text preprocessing
functions.

  src/feature_extraction.py                       Fits the TF-IDF vectorizer on
training text and transforms
training/test text.

  src/models.py                                   Defines the original scikit-learn
linear SVM model variants.

  src/thresholding.py                             Implements the original project's
threshold tuning and
threshold-based prediction
utilities.

  src/evaluation.py                               Calculates classification metrics
and saves confusion matrices.

  src/experiments.py                              Runs the original scikit-learn
experiments and saves their metrics
and model artifacts.

  src/official_models.py                          Converts labels to the sparse
format expected by LibMultiLabel,
trains the three official-method
models, and provides prediction
utilities.

  src/official_experiments.py                     Runs the LibMultiLabel experiments
using the combined training split
and untouched test split; saves
official metrics, confusion
matrices, and artifacts.

  results/metrics.csv                             Metrics from the original
experiment run.

  results/test_metrics.csv                        Test-set metrics from the original
experiment pipeline.

  results/model_artifacts.pkl                     Original vectorizer, models, and
threshold offsets used by the
original demo path.

  results/official/official_metrics.csv           Latest official LibMultiLabel
evaluation metrics.

  results/official/*confusion_matrix.png          Confusion matrices for the three
official methods.
results/official/official_model_artifacts.pkl   Official vectorizer and the three
LibMultiLabel models used by the
Streamlit app.
The data/ directory may be empty if the Hugging Face dataset cache is
used instead of saving a local dataset copy.
6. Setup and installation
Prerequisites
- Python 3.10 or later is recommended.
- Git (if cloning the repository).
- Internet access for the initial Hugging Face dataset download.
Step 1: Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd linear-classifier-reproduction
Replace the placeholder with the actual repository URL.
Step 2: Create a virtual environment
Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
If PowerShell blocks activation, you can use Command Prompt instead:
venv\Scripts\activate.bat
macOS/Linux:
python3 -m venv venv
source venv/bin/activate
Step 3: Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
If the project uses a Hugging Face account/token for higher download
limits, configure HF_TOKEN in your environment. A token is not
required for ordinary public dataset access, but unauthenticated
downloads may be rate-limited.
Important --- LibMultiLabel compatibility patches: During
development, the installed LibMultiLabel package required two local
compatibility fixes in the virtual environment:
1. In the thresholding training code, the positive-label count
   calculation was changed to:
   num_positives = np.asarray(y.sum(axis=0)).ravel()
   This addressed a dimensionality issue with the sparse label
   matrix.
2. In the installed FlatModel._to_dense_array() helper, the
   implementation did not return a value for a regular NumPy
   ndarray. The Streamlit app therefore uses a local
   official_predict_values() helper that performs the same feature
   padding/bias handling and score calculation, converting the matrix
   product to an ndarray before adding thresholds.
These are environment-level compatibility workarounds, not changes to
the paper's learning methods. They are not automatically applied by
pip install -r requirements.txt. If you create a fresh environment
and encounter the same LibMultiLabel errors, reapply the training
compatibility change to the installed package. The app's local
prediction helper avoids needing the second package patch at inference
time.

To locate the installed package file, run:
python -c "import libmultilabel.linear.linear as m; print(m.__file__)"
Keep a note of the installed LibMultiLabel version when reproducing
results.

7. How to run the experiments and regenerate results
Run commands from the repository root with the virtual environment
activated.
A. Run the original scikit-learn experiments
python -m src.experiments
This pipeline loads SCOTUS, combines train and validation, preprocesses
the text, extracts TF-IDF features, trains the original model variants,
evaluates them on the test set, and saves metrics, confusion matrices,
and model artifacts.
B. Run the official LibMultiLabel experiments
python -m src.official_experiments
This pipeline:
1. Downloads/loads SCOTUS.
2. Combines the 5,000 training and 1,400 validation examples.
3. Keeps the 1,400 test examples separate.
4. Applies the official-path preprocessing.
5. Fits TF-IDF on the training data and transforms the test data.
6. Trains OVR, Thresholding, and Cost-Sensitive models using
   LibMultiLabel.
7. Evaluates predictions on the test set.
8. Saves metrics, confusion matrices, and the official model artifacts
   under results/official/.
The official run takes longer than the original run, especially the
cost-sensitive model.
C. View the results
Open:
results/official/official_metrics.csv
You can also open the three confusion-matrix PNG files in
results/official/.

To regenerate the paper comparison, use the latest values in
official_metrics.csv and compare the Micro-F1 and Macro-F1 columns
with the paper's SCOTUS results shown in Section 4 of this README.
8. How to run the Streamlit application
The application uses the saved artifact files. Run the experiment
scripts first if these artifacts are missing or if you want to
regenerate them.

From the project root:
streamlit run app.py
Streamlit will print a local URL, usually:
http://localhost:8501
Open that URL in a browser.
Using the application
1. Paste a legal judgment into the text area.
2. Select one of the six available models.
3. Click Classify Document.
4. Read the predicted legal category.
5. Review the processing time and decision-score chart.
6. Repeat with another model using the same document if you want to
   compare predictions.
Models available in the app
  Display name                        Implementation
  Baseline Linear SVM                 Original scikit-learn

  Cost-Sensitive Linear SVM           Original scikit-learn weighted
model

  Thresholded Linear SVM              Original custom thresholding
implementation

  Official OVR (LibMultiLabel)        LibMultiLabel OVR

  Official Thresholding               LibMultiLabel thresholding
(LibMultiLabel)                     
Official Cost-Sensitive             LibMultiLabel cost-sensitive
(LibMultiLabel)
Interpreting the scores: The chart displays decision scores, not
calibrated probabilities. A score of 0.8 must not be interpreted as an
80% probability. The selected model's prediction is based on its
decision rule; the official models use their learned thresholds where
applicable.
9. Reproducing the reported comparison
To regenerate the official results from scratch:
1. Create and activate the virtual environment.
2. Install requirements.txt.
3. Apply the documented LibMultiLabel training compatibility workaround
   if the installed version requires it.
4. Run:
   python -m src.official_experiments
5. Wait for all three training procedures and test evaluation to
   finish.
6. Read results/official/official_metrics.csv.
7. Compare the Micro-F1 and Macro-F1 values with the paper's SCOTUS
   results.
The Streamlit app is a separate inference/demo step:
streamlit run app.py
It loads the saved artifacts; it does not train the models when a user
submits a document.

For a fair comparison, do not train on the test set, do not tune
thresholds using test labels, and use the same dataset configuration and
split protocol.
10. Important limitations and reproducibility notes
- The project focuses on the SCOTUS dataset/configuration, not
  every dataset evaluated in the paper.
- The original scikit-learn thresholding implementation is a project
  approximation; the LibMultiLabel thresholding routine is the
  official-method implementation used for the paper comparison.
- The displayed category is a model prediction, not legal advice or a
  legal determination.
- Decision scores are not probabilities.
- Very rare classes have few training examples, which can make
  per-class performance unstable.
- Exact numerical reproduction may depend on package versions, solver
  behavior, vectorizer defaults, preprocessing, and runtime
  environment.
- The compatibility changes described above are local environment
  workarounds. For a fully repeatable setup, record the Python and
  package versions used and preserve the patched environment or
  formalize the patch in a reproducible setup script.
11. References
- Paper: Linear Classifier: An Often-Forgotten Baseline for Text
  Classification, ACL 2023.
- Authors' code repository:
  https://github.com/JamesLYC88/text_classification_baseline_code
- Dataset: LexGLUE, SCOTUS configuration on Hugging Face:
  https://huggingface.co/datasets/coastalcph/lex_glue
- Library: LibMultiLabel:
  https://github.com/ASUS-AICS/LibMultiLabel
12. Quick command reference
  Task                                Command
  Activate environment (Windows       .\venv\Scripts\Activate.ps1
PowerShell)                         

  Install dependencies                pip install -r requirements.txt

  Run original experiments            python -m src.experiments

  Run official LibMultiLabel          python -m src.official_experiments
experiments                         

  Launch the web app                  streamlit run app.py
Locate LibMultiLabel installation   python -c "import libmultilabel.linear.linear as m; print(m.__file__)"
Project outcome: A working comparison of three original
linear-classifier baselines and three LibMultiLabel implementations on
SCOTUS, with saved test metrics, confusion matrices, serialized model
artifacts, and an interactive Streamlit legal-text classification demo.
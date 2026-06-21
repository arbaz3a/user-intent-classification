# ============================================================
# ============================================================


# ===========================================================
# ## Step 1 - Install & Import Libraries
# ===========================================================

# --- Install required packages ---


# ---- Standard Data & ML Libraries ----
import pandas as pd                          # For DataFrame operations and tabular data manipulation
import numpy as np                           # For numerical operations and array handling

# ---- HuggingFace Dataset Loader ----
from datasets import load_dataset            # To load datasets directly from HuggingFace Hub

# ---- Text Preprocessing ----
import re                                    # Regular expressions for removing special characters
import string                                # Provides string constants like punctuation characters

# ---- Feature Extraction ----
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to TF-IDF numerical vectors

# ---- Label Encoding ----
from sklearn.preprocessing import LabelEncoder               # Converts categorical text labels to integers

# ---- Train/Test Split ----
from sklearn.model_selection import train_test_split         # Splits data into training and testing sets

# ---- Machine Learning Models ----
from sklearn.linear_model import LogisticRegression          # Logistic Regression classifier
from sklearn.svm import LinearSVC                            # Support Vector Machine with linear kernel (faster than SVC)
from sklearn.tree import DecisionTreeClassifier              # Decision Tree classifier

# ---- Model Evaluation Metrics ----
from sklearn.metrics import (
    accuracy_score,          # Overall fraction of correct predictions
    f1_score,                # Harmonic mean of precision and recall
    precision_score,         # How many predicted positives are actually positive
    recall_score,            # How many actual positives were correctly predicted
    classification_report,   # Full per-class breakdown of metrics
    confusion_matrix         # Matrix showing true vs predicted label counts
)

# ---- Visualization ----
import matplotlib.pyplot as plt   # Core plotting library for charts and graphs
import seaborn as sns             # High-level wrapper over matplotlib for statistical plots
import warnings                   # To suppress non-critical warnings during model training

warnings.filterwarnings('ignore')  # Suppress convergence and other non-critical warnings for cleaner output

print("=" * 60)
print("  ALL LIBRARIES IMPORTED SUCCESSFULLY")
print("=" * 60)


# ===========================================================
# ## Step 2 - Load Dataset
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 2: LOADING DATASET FROM HUGGINGFACE")
print("=" * 60)

raw_dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",  # Dataset name on HuggingFace Hub
    split="train"                                                    # Use the training split
)

df = raw_dataset.to_pandas()

# Extract only the columns we need:
df = df[['instruction', 'intent']]  # Keep only relevant columns to reduce memory usage

print("\n--- First 5 Rows of the Dataset ---")
print(df.head())  # Show the first 5 rows so we know what the data looks like

print("\n--- Formatted Sample Table ---")
print(df[['instruction', 'intent']].head(5).to_string(index=False, justify='left'))

print(f"\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")  # Total samples and features

print("\n--- Unique Intent Labels with Counts ---")
intent_counts = df['intent'].value_counts()
print(intent_counts)
print(f"\nTotal unique intents: {df['intent'].nunique()}")  # Total number of distinct classes


# ===========================================================
# ## Step 2b — Class Distribution and Imbalance Analysis
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 2B: CLASS DISTRIBUTION & IMBALANCE ANALYSIS")
print("=" * 60)

intent_counts = df['intent'].value_counts()
plt.figure(figsize=(16, 8)) # Increased figure size to 16 by 8 for better readability

# Define gradient colormap from light blue to dark blue based on counts
import matplotlib.cm as cm # Import colormap for gradient
import matplotlib.colors as mcolors # Import colors for normalization
norm = mcolors.Normalize(vmin=intent_counts.min(), vmax=intent_counts.max()) # Normalize counts
cmap = cm.get_cmap('Blues') # Use Blues colormap
colors = [cmap(norm(value)) for value in intent_counts.values] # Generate gradient colors

# Plot intent counts as a bar chart with gradient colors
bars = intent_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.0) # Apply gradient colors

plt.ylim(940, 1010) # Set Y axis minimum to 940 and maximum to 1010

for bar in bars.patches: # Loop through each bar
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(bar.get_height())), ha='center', va='bottom', fontsize=9) # Show exact number

mean_count = intent_counts.mean()
plt.axhline(mean_count, color='red', linestyle='--', linewidth=1.5, label=f'Mean Count ({mean_count:.1f})')

# Add a second horizontal green dashed line showing the maximum count labeled as Max Count
max_count = intent_counts.max() # Get maximum count
plt.axhline(max_count, color='green', linestyle='--', linewidth=1.5, label=f'Max Count ({max_count})') # Draw green dashed line

plt.title('Class Distribution of Intent Labels', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Intent Label', fontsize=12)
plt.ylabel('Sample Count', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Class distribution chart saved as 'class_distribution.png'")

most_frequent_class = intent_counts.index[0]
most_frequent_count = intent_counts.iloc[0]
least_frequent_class = intent_counts.index[-1]
least_frequent_count = intent_counts.iloc[-1]
imbalance_ratio = most_frequent_count / least_frequent_count

print(f"\nMost frequent class : {most_frequent_class} ({most_frequent_count} samples)")
print(f"Least frequent class: {least_frequent_class} ({least_frequent_count} samples)")
print(f"Imbalance ratio     : {imbalance_ratio:.4f}")

if imbalance_ratio > 1.5:
    print("WARNING IMBALANCE DETECTED")
else:
    print("DATASET IS BALANCED")



# ===========================================================
# ## Step 3 - Data Cleaning
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 3: DATA CLEANING - DEFINING STOPWORDS")
print("=" * 60)

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
    'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
}


# ===========================================================
# ## Step 4 - Text Cleaning and Application
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 4: DEFINE AND APPLY TEXT CLEANING")
print("=" * 60)

def clean_text(text):
    """
    Applies a 4-step text cleaning pipeline to a raw customer message.

    Steps:
        1. Lowercase          — Normalize case so "Cancel" and "cancel" are treated identically.
        2. Remove punctuation — Strip symbols that add no semantic meaning.
        3. Remove stopwords   — Drop high-frequency low-signal words.
        4. Strip extra spaces — Remove leading/trailing/multiple whitespace characters.
    """
    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()                                     # Split sentence into list of words
    tokens = [word for word in tokens if word not in STOPWORDS]  # Keep only non-stopword tokens

    text = ' '.join(tokens).strip()

    return text  # Return the fully cleaned text

df['cleaned_instruction'] = df['instruction'].apply(clean_text)

print("\n--- Before Cleaning (Raw Text) ---")
print(df['instruction'].iloc[0])   # Show original text of the first sample

print("\n--- After Cleaning (Cleaned Text) ---")
print(df['cleaned_instruction'].iloc[0])   # Show cleaned version of the same sample

print("\n--- Data Cleaning Complete ---")
print(f"Total samples after cleaning: {len(df)}")  # Confirm no rows were accidentally dropped


# ===========================================================
# ## Step 5 - Label Encoding
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 5: LABEL ENCODING")
print("=" * 60)

label_encoder = LabelEncoder()  

y = label_encoder.fit_transform(df['intent'])  # y is now an integer array

print("--- Label to Integer Mapping (first 10) ---")
for idx, label in enumerate(label_encoder.classes_[:10]):   # Show first 10 for brevity
    print(f"  {idx:>3}  →  {label}")   # Format: integer → intent string

print(f"\nTotal encoded classes: {len(label_encoder.classes_)}")  # Total number of unique intents


# ===========================================================
# ## Step 6 - Train/Test Split
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 6: TRAIN / TEST SPLIT ON RAW TEXT")
print("=" * 60)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['cleaned_instruction'],  # Cleaned text strings
    y,                          # Encoded targets
    test_size=0.20,             # 20% reserved for testing
    random_state=42,            # Use random_state=42 for reproducibility (Fix 1)
    stratify=y                  # Maintain class distribution across splits
)

print(f"Training Text Set Size: {X_train_text.shape[0]} samples")   # Training text samples
print(f"Testing Text Set Size : {X_test_text.shape[0]} samples")    # Testing text samples


# ===========================================================
# ## Step 6b — TF-IDF Feature Extraction
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 6B: TF-IDF FEATURE EXTRACTION")
print("=" * 60)

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,     # Vocabulary size cap: top 5000 words by TF-IDF score
    ngram_range=(1, 2),    # Use both single words (unigrams) and word pairs (bigrams)
    sublinear_tf=True      # Apply log normalization to TF to reduce the impact of very frequent words
)

X_train = tfidf_vectorizer.fit_transform(X_train_text)

X_test = tfidf_vectorizer.transform(X_test_text)

print(f"TF-IDF Feature Matrix Shape (Train): {X_train.shape}")
print(f"TF-IDF Feature Matrix Shape (Test) : {X_test.shape}")
print(f"Vocabulary Size: {len(tfidf_vectorizer.vocabulary_)}")  # Should be ≤ 5000
sample_vocab = list(tfidf_vectorizer.vocabulary_.keys())[:20]  # First 20 vocabulary terms
print(f"Sample Vocabulary Terms: {sample_vocab}")
print(f"Feature Dimensions  : {X_train.shape[1]} features")  # TF-IDF vocabulary size


# ===========================================================
# ## Step 7 - Train 5 Models
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 7: TRAINING 5 MACHINE LEARNING MODELS")
print("=" * 60)

# ----------------------------------------------------------------
# ----------------------------------------------------------------

print("\n[1/5] Training Logistic Regression...")
logistic_regression_model = LogisticRegression(
    max_iter=1000,      # Max iterations for the optimizer to converge
    random_state=42,    # Reproducibility
    C=1.0,              # Inverse regularization strength — controls model complexity
    solver='lbfgs',     # Efficient optimizer for multi-class problems
    multi_class='auto', # Automatically selects multi-class strategy
    class_weight='balanced'  # Add balanced weights to handle class imbalance (Fix 2)
)
logistic_regression_model.fit(X_train, y_train)  # Learn weights from training data
print("  Logistic Regression trained successfully.")


# ----------------------------------------------------------------
# ----------------------------------------------------------------

print("\n[3/5] Training Support Vector Machine (LinearSVC)...")
svm_model = LinearSVC(
    C=1.0,           # Regularization parameter — trades off margin size vs training error
    max_iter=2000,   # Allow more iterations for convergence with many classes
    random_state=42, # Reproducibility (applies to random dual/primal solver choices)
    class_weight='balanced'  # Add balanced weights to handle class imbalance (Fix 2)
)
svm_model.fit(X_train, y_train)  # Learn the maximum-margin hyperplane(s)
print("  SVM (LinearSVC) trained successfully.")


# ----------------------------------------------------------------
# ----------------------------------------------------------------

print("\n[5/5] Training Decision Tree...")
decision_tree_model = DecisionTreeClassifier(
    max_depth=50,       # Maximum depth of the tree — prevents unlimited overfitting
    random_state=42,    # Fixes randomness in feature/threshold selection for reproducibility
    criterion='gini',    # Use Gini impurity to measure split quality (alternative: 'entropy')
    class_weight='balanced'  # Add balanced weights to handle class imbalance (Fix 2)
)
decision_tree_model.fit(X_train, y_train)  # Build the decision tree from training data
print("  Decision Tree trained successfully.")

print("\n" + "=" * 60)
print("  ALL 5 MODELS TRAINED SUCCESSFULLY")
print("=" * 60)


# ===========================================================
# ## Step 8 - Evaluate All Models
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 8: EVALUATING ALL MODELS")
print("=" * 60)

models_dict = {
    "Logistic Regression": logistic_regression_model,
    "SVM (LinearSVC)":     svm_model,
    "Decision Tree":       decision_tree_model
}

evaluation_results = []  # Will hold one dict per model with all metric scores

for model_name, model_object in models_dict.items():

    print(f"\n{'─' * 50}")
    print(f"  MODEL: {model_name}")
    print(f"{'─' * 50}")

    y_predicted = model_object.predict(X_test)  # Predict integer labels for all test samples

    # ---- Compute Evaluation Metrics ----
    accuracy = accuracy_score(y_test, y_predicted)

    f1 = f1_score(y_test, y_predicted, average='weighted')

    precision = precision_score(y_test, y_predicted, average='weighted')

    recall = recall_score(y_test, y_predicted, average='weighted')

    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    full_report = classification_report(
        y_test,
        y_predicted,
        target_names=label_encoder.classes_  # Decode integer labels back to intent names
    )
    print(f"\n  Full Classification Report:\n{full_report}")

    evaluation_results.append({
        "Model":     model_name,
        "Accuracy":  round(accuracy, 4),
        "F1 Score":  round(f1, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4)
    })


# ===========================================================
# ## Step 9 - Comparison Table
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 9: MODEL COMPARISON TABLE")
print("=" * 60)

comparison_df = pd.DataFrame(evaluation_results)

comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't wrap at a fixed width
pd.set_option('display.float_format', '{:.4f}'.format)  # Format floats to 4 decimal places

print("\n--- Model Performance Comparison (Sorted by Accuracy) ---\n")
print(comparison_df.to_string(index=False))  # to_string for clean tabular output in Colab

best_model_name = comparison_df.iloc[0]['Model']   # Name of the best performing model
best_model_accuracy = comparison_df.iloc[0]['Accuracy']  # Its accuracy score

print(f"\n  [BEST] BEST MODEL: {best_model_name} with Accuracy = {best_model_accuracy:.4f} ({best_model_accuracy*100:.2f}%)")

best_model_object = models_dict[best_model_name]  # Retrieve the fitted model object by name


# ===========================================================
# ## Step 10 - Confusion Matrix (Best Model)
# ===========================================================

print("\n" + "=" * 60)
print(f"  STEP 10: CONFUSION MATRIX — {best_model_name}")
print("=" * 60)

y_best_predicted = best_model_object.predict(X_test)  # Predictions on test set

conf_matrix = confusion_matrix(y_test, y_best_predicted)  # shape: (n_classes, n_classes)

num_classes = len(label_encoder.classes_)   # Total number of unique intent classes
fig_size = max(14, num_classes // 2)        # Dynamically scale figure size with class count

plt.figure(figsize=(fig_size, fig_size))    # Create a large enough figure

# cmap='Blues': Use a Blue color gradient — darker = higher count.
sns.heatmap(
    conf_matrix,
    annot=(num_classes <= 30),          # Only annotate cells if ≤ 30 classes (avoids overcrowding)
    fmt='d',                            # Display integers
    cmap='Blues',                       # Blue color scale for visual clarity
    xticklabels=label_encoder.classes_, # Column labels = predicted intent names
    yticklabels=label_encoder.classes_, # Row labels = true intent names
    linewidths=0.5                      # Add thin grid lines between cells for readability
)

plt.title(f'Confusion Matrix — {best_model_name}', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Intent Label', fontsize=12)   # X-axis: what the model predicted
plt.ylabel('True Intent Label', fontsize=12)        # Y-axis: the actual ground truth label
plt.xticks(rotation=90, fontsize=8)                 # Rotate x-axis labels to avoid overlap
plt.yticks(rotation=0, fontsize=8)                  # Keep y-axis labels horizontal
plt.tight_layout()                                  # Auto-adjust margins to prevent label clipping
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')  # Save to disk
plt.show()                                          # Display the plot in Colab
print("  Confusion matrix saved as 'confusion_matrix.png'")


# ===========================================================
# ## Step 11 - Accuracy Bar Chart (All Models)
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 11: ACCURACY BAR CHART — ALL MODELS")
print("=" * 60)

model_names = comparison_df['Model'].tolist()       # List of model names (sorted by accuracy)
model_accuracies = comparison_df['Accuracy'].tolist()  # Corresponding accuracy scores

bar_colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']  # Blue, Green, Red, Purple, Orange

plt.figure(figsize=(12, 6))  # Wide figure to accommodate model name labels

bars = plt.bar(
    model_names,      # X-axis: model names
    model_accuracies, # Y-axis: accuracy values
    color=bar_colors, # Assign a different color to each bar
    edgecolor='black',# Black border around each bar for visual crispness
    linewidth=1.2,    # Border thickness
    width=0.6         # Bar width (slightly narrower than default for cleaner look)
)

for bar_obj, acc_value in zip(bars, model_accuracies):
    plt.text(
        bar_obj.get_x() + bar_obj.get_width() / 2,  # Horizontal center of the bar
        bar_obj.get_height() + 0.005,               # Slightly above the top of the bar
        f'{acc_value:.4f}',                         # Accuracy formatted to 4 decimal places
        ha='center',                                # Horizontally centered text
        va='bottom',                                # Anchored to bottom of text
        fontsize=11,                                # Readable font size
        fontweight='bold'                           # Bold for emphasis
    )

plt.axhline(
    y=max(model_accuracies),  # Draw line at the height of the best score
    color='red',
    linestyle='--',
    linewidth=1.5,
    alpha=0.7,
    label=f'Best Accuracy ({max(model_accuracies):.4f})'  # Label for legend
)

plt.title('Model Accuracy Comparison — User Intent Classification', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Machine Learning Model', fontsize=12)  # X-axis label
plt.ylabel('Accuracy Score (0 to 1)', fontsize=12)  # Y-axis label

y_min = max(0, min(model_accuracies) - 0.05)  # Don't go below 0
plt.ylim(y_min, 1.05)                          # Add headroom above 1.0 for annotations

plt.xticks(rotation=15, ha='right', fontsize=10)  # Rotate labels to avoid overlap
plt.yticks(fontsize=10)                           # Y-axis tick labels
plt.legend(fontsize=10)                           # Show the reference line label
plt.grid(axis='y', linestyle='--', alpha=0.4)     # Horizontal grid lines for readability
plt.tight_layout()                                # Prevent label/title clipping
plt.savefig('accuracy_bar_chart.png', dpi=150, bbox_inches='tight')  # Save plot to disk
plt.show()                                        # Display in Colab
print("  Accuracy bar chart saved as 'accuracy_bar_chart.png'")


# ===========================================================
# ## Step 12 - Live Demo: predict_intent() Function
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 12: LIVE DEMO — predict_intent() FUNCTION")
print("=" * 60)


def predict_intent(message):
    """
    Predicts the intent of a customer support message using the best trained model.

    Pipeline:
        1. Clean the input message using the SAME cleaning steps as training data.
           (Critical: any mismatch between training and serving preprocessing causes garbage predictions.)
        2. Transform it using the SAME fitted TF-IDF vectorizer.
           (We transform(), NOT fit_transform() — we don't re-learn vocabulary at inference time.)
        3. Predict the integer label using the best model.
        4. Decode the predicted integer back to the original human-readable intent string
           using the SAME fitted LabelEncoder.

    Args:
        message (str): A raw customer support message or query.

    Returns:
        str: The predicted intent label (human-readable string, e.g., "cancel_order").

    Example:
        >>> predict_intent("I want to cancel my order")
        Predicted Intent: cancel_order
    """

    cleaned_message = clean_text(message)  # Lowercase, remove punctuation, remove stopwords

    message_vector = tfidf_vectorizer.transform([cleaned_message])  # Shape: (1, 5000)

    predicted_label_integer = best_model_object.predict(message_vector)[0]  # e.g., 3

    predicted_intent_string = label_encoder.inverse_transform([predicted_label_integer])[0]  # e.g., "cancel_order"

    print(f"\n  Input Message   : \"{message}\"")
    print(f"  Cleaned Message : \"{cleaned_message}\"")
    print(f"  Predicted Intent: {predicted_intent_string}")
    print(f"  {'─' * 45}")

    return predicted_intent_string   # Also return the string so it can be used programmatically


# ---- Test the predict_intent() function with 3 real-world example messages ----

print("\n--- Running Live Demo Predictions ---\n")

# Test Case 1: User wants to cancel their order
result_1 = predict_intent("I want to cancel my order")  # Expected: something like 'cancel_order'

result_2 = predict_intent("Where is my refund?")  # Expected: something like 'track_refund'

result_3 = predict_intent("My product arrived damaged")  # Expected: something like 'return_item' or 'complaint'

print("\n" + "=" * 60)
print("  ALL DEMO PREDICTIONS COMPLETE")
print("  Summary of Results:")
print(f"    'I want to cancel my order'  → {result_1}")
print(f"    'Where is my refund?'        → {result_2}")
print(f"    'My product arrived damaged' → {result_3}")
print("=" * 60)

print("\n" + "=" * 60)
print("  [OK] USER INTENT CLASSIFICATION PROJECT COMPLETE!")
print(f"  Best Model : {best_model_name}")
print(f"  Accuracy   : {best_model_accuracy:.4f} ({best_model_accuracy * 100:.2f}%)")
print("=" * 60)



# ===========================================================
# ## Step 13 - Overfitting Detection (Train vs Test Accuracy)
# ===========================================================

print("=" * 60)
print("  STEP 13: OVERFITTING DETECTION (TRAIN vs TEST ACCURACY)")
print("=" * 60)

overfit_results = []   # Will hold one dict per model with diagnosis info

for model_name, model_obj in models_dict.items():   # Loop over all 5 trained models

    y_train_pred = model_obj.predict(X_train)                      # Predictions on training data
    train_acc    = accuracy_score(y_train, y_train_pred)           # Training set accuracy

    y_test_pred = model_obj.predict(X_test)                        # Predictions on test data
    test_acc    = accuracy_score(y_test, y_test_pred)              # Test set accuracy

    gap = train_acc - test_acc   # Positive = train better than test

    if gap > 0.10:               # More than 10% accuracy drop -> overfitting
        status = "OVERFITTING DETECTED"
    elif gap < -0.05:            # Test accuracy > train by >5% -> underfitting signal
        status = "UNDERFITTING DETECTED"
    else:                        # Within acceptable bounds -> healthy generalisation
        status = "GOOD FIT"

    print("\n  " + model_name)
    print("    Train Accuracy : " + "{:.4f}".format(train_acc))
    print("    Test  Accuracy : " + "{:.4f}".format(test_acc))
    print("    Gap            : " + "{:+.4f}".format(gap))   # +/- sign shows direction
    print("    Status         : " + status)

    overfit_results.append({
        "Model":          model_name,
        "Train Accuracy": round(train_acc, 4),  # Rounded to 4 decimal places
        "Test Accuracy":  round(test_acc, 4),   # Rounded to 4 decimal places
        "Gap":            round(gap, 4),         # Rounded to 4 decimal places
        "Status":         status                 # Human-readable diagnosis
    })

overfit_df = pd.DataFrame(overfit_results)

overfit_df = overfit_df.sort_values(by="Gap", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("  OVERFITTING SUMMARY TABLE (sorted by gap, highest first)")
print("=" * 60)
print(overfit_df.to_string(index=False))   # Print without row index for cleanliness

fig, ax = plt.subplots(figsize=(13, 6))   # Wide figure to fit 5 model groups comfortably

n_models    = len(overfit_df)          # Total number of models (5)
x_positions = np.arange(n_models)     # Integer positions [0,1,2,3,4] for each model group
bar_width   = 0.35                    # Width of each individual bar within a group

train_bars = ax.bar(
    x_positions - bar_width / 2,      # Shift left by half bar_width
    overfit_df["Train Accuracy"],      # Bar heights = train accuracy values
    width=bar_width,                   # Bar width
    label="Train Accuracy",           # Legend entry
    color="#42A5F5",                  # Light blue = training performance
    edgecolor="black",                # Black border for visual clarity
    linewidth=0.8
)

test_bars = ax.bar(
    x_positions + bar_width / 2,      # Shift right by half bar_width
    overfit_df["Test Accuracy"],       # Bar heights = test accuracy values
    width=bar_width,                   # Same width as train bars
    label="Test Accuracy",            # Legend entry
    color="#EF5350",                  # Red-orange = test/real-world performance
    edgecolor="black",
    linewidth=0.8
)

for bar in train_bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,   # Horizontal centre of bar
        bar.get_height() + 0.005,            # Just above the bar top
        "{:.3f}".format(bar.get_height()),   # Value to 3 decimal places
        ha="center", va="bottom",
        fontsize=8, fontweight="bold"
    )

for bar in test_bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        "{:.3f}".format(bar.get_height()),
        ha="center", va="bottom",
        fontsize=8, fontweight="bold"
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(overfit_df["Model"].tolist(), rotation=12, ha="right", fontsize=10)

ax.set_title(
    "Train vs Test Accuracy — Overfitting Detection",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Accuracy Score (0-1)", fontsize=12)

y_min_all = min(overfit_df["Test Accuracy"].min(), overfit_df["Train Accuracy"].min())
y_floor   = max(0, y_min_all - 0.05)
ax.set_ylim(y_floor, 1.07)   # Upper headroom for value annotations

ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)   # Horizontal grid lines
plt.tight_layout()
plt.savefig("overfitting_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nOverfitting chart saved as 'overfitting_chart.png'")



# ===========================================================
# ## Step 14 - Fix Decision Tree Overfitting (max_depth=10)
# ===========================================================

print("=" * 60)
print("  STEP 14: FIXING DECISION TREE OVERFITTING (max_depth=10)")
print("=" * 60)


dt_fixed = DecisionTreeClassifier(
    max_depth=10,       # Key change: reduced from 50 -> forces simpler generalising splits
    criterion="gini",   # Same split criterion as the original tree
    random_state=42,    # Same seed for a fair comparison
    class_weight='balanced'  # Use balanced weights for consistency with base models (Fix 2)
)
dt_fixed.fit(X_train, y_train)   # Fit on same training data as the original tree

dt_original     = models_dict["Decision Tree"]                              # Original model object
orig_train_acc  = accuracy_score(y_train, dt_original.predict(X_train))    # Original train accuracy
orig_test_acc   = accuracy_score(y_test,  dt_original.predict(X_test))     # Original test accuracy
orig_gap        = orig_train_acc - orig_test_acc                            # Original overfit gap

fixed_train_acc = accuracy_score(y_train, dt_fixed.predict(X_train))       # Fixed train accuracy
fixed_test_acc  = accuracy_score(y_test,  dt_fixed.predict(X_test))        # Fixed test accuracy
fixed_gap       = fixed_train_acc - fixed_test_acc                          # Fixed overfit gap

print("\n--- Decision Tree: Original (max_depth=50) ---")
print("  Train Accuracy : " + "{:.4f}".format(orig_train_acc))
print("  Test  Accuracy : " + "{:.4f}".format(orig_test_acc))
orig_label = "[OVERFITTING]" if orig_gap > 0.10 else "[OK]"
print("  Gap            : " + "{:+.4f}  {}".format(orig_gap, orig_label))

print("\n--- Decision Tree: Fixed (max_depth=10) ---")
print("  Train Accuracy : " + "{:.4f}".format(fixed_train_acc))
print("  Test  Accuracy : " + "{:.4f}".format(fixed_test_acc))
fixed_label = "[OVERFITTING]" if fixed_gap > 0.10 else "[OK]"
print("  Gap            : " + "{:+.4f}  {}".format(fixed_gap, fixed_label))

test_improvement = fixed_test_acc - orig_test_acc   # Positive = fixed model better on test
gap_reduction    = orig_gap - fixed_gap             # Positive = overfit gap got smaller

print("\n--- Improvement Summary ---")
print("  Test Accuracy Change  : " + "{:+.4f}".format(test_improvement))
print("  Gap Reduced By        : " + "{:+.4f}".format(gap_reduction))
conclusion = "generalises better" if test_improvement >= 0 else "trades some accuracy for robustness"
print("  Conclusion: The fixed tree " + conclusion + ".")



# ===========================================================
# ## Step 15 - 5-Fold Cross-Validation (All Models)
# ===========================================================

print("=" * 60)
print("  STEP 15: 5-FOLD CROSS-VALIDATION (ALL MODELS)")
print("=" * 60)

from sklearn.model_selection import cross_val_score   # k-fold CV utility from sklearn


cv_results     = []   # Accumulate results for the comparison table and chart
cv_scores_dict = {}   # Store raw arrays for error bars in the chart

for model_name, model_obj in models_dict.items():   # Loop over all 5 trained models

    scores = cross_val_score(
        model_obj,          # Fitted model object (will be cloned internally)
        X_train,            # Training features only — never X_test!
        y_train,            # Training labels only
        cv=5,               # Number of folds
        scoring="accuracy", # Metric to compute per fold
        n_jobs=-1           # Parallelise across all CPU cores
    )

    mean_cv = scores.mean()   # Average accuracy across 5 folds
    std_cv  = scores.std()    # Standard deviation — measures stability

    print("\n  " + model_name)
    print("    Fold Scores  : " + str([round(s, 4) for s in scores]))
    print("    Mean CV Score: " + "{:.4f}".format(mean_cv))
    stability_note = "<- HIGH VARIANCE: UNSTABLE" if std_cv > 0.02 else "<- LOW VARIANCE: STABLE"
    print("    Std Deviation: " + "{:.4f}  {}".format(std_cv, stability_note))

    cv_scores_dict[model_name] = scores   # Save raw array for error bars in chart

    cv_results.append({
        "Model":    model_name,
        "CV Mean":  round(mean_cv, 4),      # Central estimate of performance
        "CV Std":   round(std_cv, 4),       # Stability measure
        "Min Fold": round(scores.min(), 4), # Worst-case fold performance
        "Max Fold": round(scores.max(), 4)  # Best-case fold performance
    })

cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values("CV Mean", ascending=False).reset_index(drop=True)  # Best first

print("\n" + "=" * 60)
print("  5-FOLD CROSS-VALIDATION SUMMARY (sorted by CV Mean)")
print("=" * 60)
print(cv_df.to_string(index=False))

best_cv_score = cv_df.iloc[0]["CV Mean"]   # Highest mean CV accuracy
best_cv_model = cv_df.iloc[0]["Model"]     # Model that achieved it
print("\n  Best CV Score: " + "{:.4f}  ({})".format(best_cv_score, best_cv_model))

fig, ax = plt.subplots(figsize=(12, 6))

cv_model_names = cv_df["Model"].tolist()   # Model names sorted by CV mean
cv_means       = cv_df["CV Mean"].tolist() # Mean CV accuracy per model
cv_stds        = cv_df["CV Std"].tolist()  # Std deviation per model

cv_bar_colors = ["#26A69A", "#AB47BC", "#FF7043", "#5C6BC0", "#66BB6A"]

bars = ax.bar(
    cv_model_names,      # X-axis: model names
    cv_means,            # Bar heights: mean CV accuracy
    yerr=cv_stds,        # Error bars: +/- std deviation from bar top
    capsize=6,           # Horizontal caps on error bars for readability
    color=cv_bar_colors, # Distinct colour per bar
    edgecolor="black",   # Black border
    linewidth=0.9,
    width=0.55,          # Slightly narrower than default
    error_kw={
        "elinewidth": 1.8,   # Error bar line thickness
        "ecolor": "black",   # Error bar colour
        "capthick": 1.8      # Cap line thickness
    }
)

for bar_obj, mean_val, std_val in zip(bars, cv_means, cv_stds):
    ax.text(
        bar_obj.get_x() + bar_obj.get_width() / 2,   # Horizontal centre
        bar_obj.get_height() + std_val + 0.008,       # Above error bar cap
        "{:.4f}".format(mean_val),                    # Mean accuracy label
        ha="center", va="bottom",
        fontsize=9, fontweight="bold"
    )

ax.set_title(
    "5-Fold Cross-Validation Mean Accuracy (Error Bars = Std Deviation)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("CV Mean Accuracy (0-1)", fontsize=12)

y_floor_cv = max(0, min(cv_means) - 0.08)   # Lower bound: a bit below minimum
ax.set_ylim(y_floor_cv, 1.08)               # Upper headroom for annotations

ax.set_xticklabels(cv_model_names, rotation=12, ha="right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("cross_validation_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nCross-validation chart saved as 'cross_validation_chart.png'")


# ===========================================================
# ## Step 15b — Standard Deviation Deep Dive
# ===========================================================

print("\n" + "=" * 60)
print("  STEP 15B: STANDARD DEVIATION DEEP DIVE")
print("=" * 60)

def get_stability_rating(std):
    if std < 0.01:
        return "VERY STABLE"
    elif 0.01 <= std <= 0.02:
        return "STABLE"
    elif 0.02 < std <= 0.03:
        return "MODERATE"
    else:
        return "UNSTABLE"

std_results = []
for res in cv_results:
    std_results.append({
        "Model": res["Model"],
        "CV Mean": res["CV Mean"],
        "CV Std": res["CV Std"],
        "Stability Rating": get_stability_rating(res["CV Std"])
    })

std_df = pd.DataFrame(std_results)
std_df = std_df.sort_values("CV Std", ascending=True).reset_index(drop=True)

print("\n--- Detailed Stability Analysis Table ---")
print(std_df.to_string(index=False))

plt.figure(figsize=(10, 5))
sorted_models = std_df["Model"].tolist()
sorted_stds = std_df["CV Std"].tolist()

bar_colors_std = ["green" if std < 0.02 else "red" for std in sorted_stds]

plt.barh(sorted_models, sorted_stds, color=bar_colors_std, edgecolor="black", height=0.55)

plt.axvline(0.02, color="red", linestyle="--", linewidth=1.5, label="Stability Threshold (0.02)")

plt.title("Model Cross-Validation Standard Deviation (Lower is Better)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Standard Deviation (CV Std)", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.legend(fontsize=10)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig("std_deviation_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nStandard deviation chart saved as 'std_deviation_chart.png'")

print("\n--- Stability & Deployment Reliability Explanation ---")
print("- Low Standard Deviation (< 0.02, rated STABLE/VERY STABLE):")
print("  Indicates that the model performs consistently across different folds of training data.")
print("  This predictability implies high reliability, as the model's production behavior will closely align with local validation metrics.")
print("- High Standard Deviation (>= 0.02, rated MODERATE/UNSTABLE):")
print("  Indicates that the model is sensitive to the specific subset of data it trains on.")
print("  This instability poses a deployment risk, as performance might drop unexpectedly on newly seen real-world distributions.")



# ===========================================================
# ## Step 16 - Hyperparameter Tuning (Best Model Only)
# ===========================================================

print("=" * 60)
print("  STEP 16: HYPERPARAMETER TUNING — " + best_model_name)
print("=" * 60)


if best_model_name == "Logistic Regression":
    param_name   = "C (Regularisation)"     # Human-readable label for the chart axis
    param_values = [0.01, 0.1, 1, 10, 100]  # Span 4 orders of magnitude

    cv_param_scores = []   # Store mean CV accuracy for each parameter value
    for c_val in param_values:
        model_variant = LogisticRegression(
            C=c_val,           # Only C changes — all other settings fixed
            max_iter=1000,     # Sufficient iterations for convergence
            random_state=42,   # Reproducibility
            solver="lbfgs",    # Same solver as original model
            multi_class="auto"
        )
        scores = cross_val_score(model_variant, X_train, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1)
        cv_param_scores.append(scores.mean())   # Store mean fold accuracy
        print("  C=" + str(c_val) + "  ->  Mean CV Accuracy: " + "{:.4f}".format(scores.mean()))

elif best_model_name == "SVM (LinearSVC)":
    param_name   = "C (Penalty)"
    param_values = [0.01, 0.1, 1, 10]

    cv_param_scores = []
    for c_val in param_values:
        model_variant = LinearSVC(
            C=c_val,         # Vary penalty hyperparameter
            max_iter=2000,   # Sufficient iterations
            random_state=42
        )
        scores = cross_val_score(model_variant, X_train, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1)
        cv_param_scores.append(scores.mean())
        print("  C=" + str(c_val) + "  ->  Mean CV Accuracy: " + "{:.4f}".format(scores.mean()))


elif best_model_name == "Decision Tree":
    param_name   = "max_depth"
    param_values = [5, 10, 20, 30]

    cv_param_scores = []
    for depth_val in param_values:
        model_variant = DecisionTreeClassifier(
            max_depth=depth_val,  # Vary depth hyperparameter
            criterion="gini",
            random_state=42
        )
        scores = cross_val_score(model_variant, X_train, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1)
        cv_param_scores.append(scores.mean())
        print("  max_depth=" + str(depth_val) + "  ->  Mean CV Accuracy: " + "{:.4f}".format(scores.mean()))


else:
    print("  No tuning logic defined for '" + best_model_name + "'. Skipping.")
    param_name    = "N/A"
    param_values  = []
    cv_param_scores = []

if cv_param_scores:
    best_param_idx   = int(np.argmax(cv_param_scores))    # Index of highest CV score
    best_param_value = param_values[best_param_idx]        # The best parameter value
    best_param_score = cv_param_scores[best_param_idx]     # Its mean CV accuracy

    print("\n  Best " + param_name + ": " + str(best_param_value))
    print("  Best CV Accuracy   : " + "{:.4f}".format(best_param_score))

    plt.figure(figsize=(10, 5))

    plt.plot(
        [str(v) for v in param_values],   # X-axis: parameter values as strings
        cv_param_scores,                  # Y-axis: mean CV accuracy per value
        marker="o",                       # Circle markers at each data point
        color="#1565C0",                  # Dark blue line
        linewidth=2.2,
        markersize=9,                     # Marker size
        markerfacecolor="#FFCA28",        # Yellow fill for markers
        markeredgecolor="#1565C0",        # Blue border on markers
        markeredgewidth=1.5
    )

    for i, (pv, score) in enumerate(zip(param_values, cv_param_scores)):
        plt.annotate(
            "{:.4f}".format(score),   # Score to 4 decimal places
            xy=(i, score),            # Point position
            xytext=(0, 12),           # 12 pixels above the marker
            textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold"
        )

    plt.axvline(
        x=best_param_idx,                              # X position of best param
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="Best " + param_name + " = " + str(best_param_value)
    )

    plt.title(
        "Hyperparameter Tuning — " + best_model_name + " (" + param_name + ")",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel("5-Fold CV Mean Accuracy", fontsize=12)

    y_floor_ht = max(0, min(cv_param_scores) - 0.05)
    plt.ylim(y_floor_ht, 1.06)

    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("hyperparameter_tuning.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nHyperparameter tuning chart saved as 'hyperparameter_tuning.png'")



# ===========================================================
# ## Step 17 - Final Summary Report
# ===========================================================

print("=" * 70)
print("  STEP 17: FINAL PROJECT SUMMARY REPORT")
print("=" * 70)

total_samples = len(df)                  # Total number of examples in the dataset
total_intents = df["intent"].nunique()   # Total number of unique intent classes


overfit_summary = dict(zip(overfit_df["Model"], overfit_df["Status"]))


if param_values:   # Guard: only if tuning was actually run in Step 16
    hp_summary = "{} = {}  (CV: {:.4f})".format(param_name, best_param_value, best_param_score)
else:
    hp_summary = "N/A (tuning skipped)"   # Fallback string

acc_pct = "{:.2f}%".format(best_model_accuracy * 100)          # e.g. "93.47%"
acc_str = "{:.4f}  ({})".format(best_model_accuracy, acc_pct)  # e.g. "0.9347  (93.47%)"
cv_str  = "{:.4f}  ({})".format(best_cv_score, best_cv_model)  # e.g. "0.9201  (SVM...)"

print("+" + "=" * 66 + "+")
print("|{:^66}|".format("USER INTENT CLASSIFICATION - FINAL SUMMARY REPORT"))
print("+" + "=" * 66 + "+")
print("|" + " " * 66 + "|")
print("|  DATASET                                                         |")
print("|    Total Samples        : {:<6}                                  |".format(total_samples))
print("|    Unique Intent Classes: {:<6}                                  |".format(total_intents))
print("|    Source               : bitext/customer-support-llm-chatbot    |")
print("|" + " " * 66 + "|")
print("+" + "-" * 66 + "+")
print("|" + " " * 66 + "|")
print("|  BEST MODEL                                                      |")
print("|    Name     : {:<50}|".format(best_model_name))
print("|    Accuracy : {:<50}|".format(acc_str))
print("|" + " " * 66 + "|")
print("+" + "-" * 66 + "+")
print("|" + " " * 66 + "|")
print("|  OVERFITTING ANALYSIS                                            |")
for mname, mstatus in overfit_summary.items():   # One row per model
    row = "|    {:<26}: {:<34}|".format(mname, mstatus)   # Fixed-width alignment
    print(row)
print("|" + " " * 66 + "|")
print("+" + "-" * 66 + "+")
print("|" + " " * 66 + "|")
print("|  CROSS-VALIDATION (5-FOLD)                                       |")
print("|    Best CV Score : {:<47}|".format(cv_str))
print("|" + " " * 66 + "|")
print("+" + "-" * 66 + "+")
print("|" + " " * 66 + "|")
print("|  HYPERPARAMETER TUNING                                           |")
print("|    Best Setting: {:<48}|".format(hp_summary))
print("|" + " " * 66 + "|")
print("+" + "-" * 66 + "+")
print("|" + " " * 66 + "|")
print("|  RECOMMENDATIONS FOR FUTURE IMPROVEMENT                          |")
print("|" + " " * 66 + "|")
print("|  1. ENSEMBLE METHODS                                             |")
print("|     Replace individual models with Random Forest or XGBoost.    |")
print("|     Ensembles combine multiple weak learners -> higher accuracy, |")
print("|     lower variance, and better handling of class imbalance.      |")
print("|" + " " * 66 + "|")
print("|  2. DEEP LEARNING (BERT / DistilBERT)                           |")
print("|     Replace TF-IDF + classical ML with a pre-trained transformer.|")
print("|     BERT understands word CONTEXT - critical for nuanced intent  |")
print("|     detection. Fine-tuning DistilBERT typically achieves 95%+   |")
print("|     accuracy vs ~85-90% for classical ML approaches.            |")
print("|" + " " * 66 + "|")
print("|  3. HANDLE CLASS IMBALANCE (SMOTE / CLASS WEIGHTS)              |")
print("|     If some intents have far fewer training samples, models      |")
print("|     will be biased toward the majority class.                    |")
print("|     Use class_weight='balanced' in sklearn or generate synthetic |")
print("|     minority-class samples with SMOTE (imbalanced-learn lib).   |")
print("|" + " " * 66 + "|")
print("+" + "=" * 66 + "+")
print("")
print("  ALL STEPS COMPLETE - Notebook ready for Google Colab submission.")
print("=" * 70)

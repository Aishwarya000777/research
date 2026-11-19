
from datasets import load_dataset
import pandas as pd


def load_scotus_dataset():

    print("Downloading SCOTUS dataset...")

    dataset = load_dataset("coastalcph/lex_glue", "scotus")

    train_df = pd.DataFrame(dataset["train"])
    validation_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    print("\nDataset downloaded successfully!")

    print("\n========== DATASET SUMMARY ==========")

    print("Train shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)
    print("Test shape:", test_df.shape)

    print("\nColumns:", train_df.columns.tolist())

    # Label metadata
    print("\n========== DATASET FEATURES ==========")

    print(dataset["train"].features)

    label_feature = dataset["train"].features["label"]

    if hasattr(label_feature, "names") and label_feature.names:
        print("\nLabel mapping:")
        for i, name in enumerate(label_feature.names):
            print(f"{i}: {name}")
    else:
        print("\nNo label names available in dataset metadata.")

    # Label distribution
    for split_name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]:
        print(f"\n{split_name} LABEL DISTRIBUTION:")
        print(df["label"].value_counts().sort_index())

    # Sample document
    sample = train_df.iloc[0]

    print("\n========== SAMPLE DOCUMENT ==========")
    print(str(sample["text"])[:500], "...")

    print("\nSample label:", sample["label"])

    return train_df, validation_df, test_df


if __name__ == "__main__":

    train, validation, test = load_scotus_dataset()

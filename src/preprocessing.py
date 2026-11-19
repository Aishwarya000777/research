
import re


def preprocess_text(text):
    """
    Basic text normalization for legal documents.
    """

    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation while retaining useful characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    sample = """
    THE SUPREME COURT held that the judgment
    violated constitutional provisions!!!
    """

    print("Original text:")
    print(sample)

    print("\nPreprocessed text:")
    print(preprocess_text(sample))

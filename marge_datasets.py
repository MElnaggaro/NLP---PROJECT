import os
import json
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.join(os.getcwd(), "reddit_project_data")

DEMOCRATS_PATH = os.path.join(
    BASE_DIR,
    "subreddit-democrats",
    "utterances.jsonl"
)

REPUBLICAN_PATH = os.path.join(
    BASE_DIR,
    "subreddit-Republican",
    "utterances.jsonl"
)

SAMPLE_SIZE = 50000
RANDOM_STATE = 42
OUTPUT_FILE = "political_reddit_100k.csv"


# =========================
# HELPER FUNCTION
# =========================
def load_clean_texts(file_path):
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}"):
            try:
                item = json.loads(line)
                text = item.get("text", "")

                if not isinstance(text, str):
                    continue

                text = text.strip()

                # Remove invalid Reddit texts
                if (
                    text == ""
                    or text.lower() in ["[deleted]", "[removed]", "deleted", "removed"]
                    or len(text) < 3
                ):
                    continue

                texts.append(text)

            except Exception:
                continue

    return texts


# =========================
# LOAD DATA
# =========================
print("Loading Democrats dataset...")
democrats_texts = load_clean_texts(DEMOCRATS_PATH)

print(f"Valid Democrats samples found: {len(democrats_texts):,}")

print("\nLoading Republicans dataset...")
republican_texts = load_clean_texts(REPUBLICAN_PATH)

print(f"Valid Republicans samples found: {len(republican_texts):,}")


# =========================
# CHECK SAMPLE SIZE
# =========================
if len(democrats_texts) < SAMPLE_SIZE:
    raise ValueError(
        f"Not enough Democrats samples. Found {len(democrats_texts)}"
    )

if len(republican_texts) < SAMPLE_SIZE:
    raise ValueError(
        f"Not enough Republicans samples. Found {len(republican_texts)}"
    )


# =========================
# RANDOM SAMPLING
# =========================
democrats_df = pd.DataFrame({
    "text": democrats_texts,
    "label": 0
}).sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

republicans_df = pd.DataFrame({
    "text": republican_texts,
    "label": 1
}).sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)


# =========================
# MERGE + SHUFFLE
# =========================
final_df = pd.concat(
    [democrats_df, republicans_df],
    ignore_index=True
).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# =========================
# SAVE
# =========================
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 50)
print("Dataset created successfully!")
print(f"Final dataset shape: {final_df.shape}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 50)

print("\nClass distribution:")
print(final_df["label"].value_counts())
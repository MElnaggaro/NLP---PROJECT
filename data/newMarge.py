import os
import json
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
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

SAMPLE_SIZE_PER_CLASS = 50000
RANDOM_STATE = 42
OUTPUT_FILE = "political_reddit_context_100k.csv"


# ==========================================
# VALIDATION
# ==========================================
INVALID_TEXTS = {
    "",
    "[deleted]",
    "[removed]",
    "deleted",
    "removed"
}


# ==========================================
# LOADER
# ==========================================
def load_reddit_rows(file_path, label):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(
            f,
            desc=f"Reading {os.path.basename(file_path)}"
        ):
            try:
                item = json.loads(line)

                text = item.get("text", "")
                if not isinstance(text, str):
                    continue

                text = text.strip()

                if (
                    text.lower() in INVALID_TEXTS
                    or len(text) < 3
                ):
                    continue

                row = {
                    "id": item.get("id"),
                    "root": item.get("root"),
                    "reply_to": item.get("reply_to"),
                    "text": text,
                    "label": label
                }

                # make sure critical fields exist
                if (
                    row["id"] is None
                    or row["root"] is None
                ):
                    continue

                rows.append(row)

            except Exception:
                continue

    return rows


# ==========================================
# LOAD DATA
# ==========================================
print("Loading Democrats...")
democrats_rows = load_reddit_rows(
    DEMOCRATS_PATH,
    label=0
)

print(f"Valid Democrats rows: {len(democrats_rows):,}")

print("\nLoading Republicans...")
republican_rows = load_reddit_rows(
    REPUBLICAN_PATH,
    label=1
)

print(f"Valid Republicans rows: {len(republican_rows):,}")


# ==========================================
# CHECK SUFFICIENCY
# ==========================================
if len(democrats_rows) < SAMPLE_SIZE_PER_CLASS:
    raise ValueError(
        f"Need {SAMPLE_SIZE_PER_CLASS} Democrats, found {len(democrats_rows)}"
    )

if len(republican_rows) < SAMPLE_SIZE_PER_CLASS:
    raise ValueError(
        f"Need {SAMPLE_SIZE_PER_CLASS} Republicans, found {len(republican_rows)}"
    )


# ==========================================
# SAMPLE
# ==========================================
democrats_df = pd.DataFrame(democrats_rows).sample(
    n=SAMPLE_SIZE_PER_CLASS,
    random_state=RANDOM_STATE
)

republicans_df = pd.DataFrame(republican_rows).sample(
    n=SAMPLE_SIZE_PER_CLASS,
    random_state=RANDOM_STATE
)


# ==========================================
# MERGE + SHUFFLE
# ==========================================
final_df = pd.concat(
    [democrats_df, republicans_df],
    ignore_index=True
)

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ==========================================
# SAVE
# ==========================================
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("NEW CONTEXT-AWARE DATASET CREATED SUCCESSFULLY")
print("=" * 60)
print(f"Shape: {final_df.shape}")
print(f"Saved as: {OUTPUT_FILE}")

print("\nColumns:")
print(final_df.columns.tolist())

print("\nClass distribution:")
print(final_df["label"].value_counts())
"""Builder script that emits opti.ipynb with the full implementation."""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"D:\PROJECTS\NLP - PROJECT\opti.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

# ============================================================
# TITLE
# ============================================================
cells.append(md(
    "# Context-Aware Political Reddit Classification (CNN-LSTM)\n"
    "\n"
    "Improvement project over the Shirley Cheng (Stanford CS224N) baseline.\n"
    "Core idea: enrich short Reddit comments with their parent post text using a `[SEP]` marker, "
    "embed with CBOW Word2Vec trained on the in-domain training corpus, and classify with a CNN-LSTM in Keras.\n"
    "\n"
    "**Labels:** 0 = Democrat, 1 = Republican.\n"
))

# ============================================================
# SECTION 1 — Imports + Config
# ============================================================
cells.append(md("## 1. Imports, Config, and Reproducibility\n"))

cells.append(code(
    "import os\n"
    "import random\n"
    "import pickle\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "import nltk\n"
    "from nltk.corpus import stopwords\n"
    "from nltk.tokenize import word_tokenize\n"
    "from nltk.stem import PorterStemmer\n"
    "\n"
    "from gensim.models import Word2Vec\n"
    "\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import (\n"
    "    accuracy_score, precision_score, recall_score, f1_score,\n"
    "    confusion_matrix, classification_report,\n"
    ")\n"
    "\n"
    "import tensorflow as tf\n"
    "from tensorflow.keras.preprocessing.text import Tokenizer\n"
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n"
    "from tensorflow.keras.models import Model, load_model\n"
    "from tensorflow.keras.layers import (\n"
    "    Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense,\n"
    "    Dropout, SpatialDropout1D,\n"
    ")\n"
    "from tensorflow.keras.optimizers import Adam\n"
    "from tensorflow.keras.callbacks import (\n"
    "    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,\n"
    ")\n"
    "\n"
    "for resource in ['punkt', 'punkt_tab', 'stopwords']:\n"
    "    try:\n"
    "        nltk.download(resource, quiet=True)\n"
    "    except Exception as e:\n"
    "        print(f'Failed to download {resource}: {e}')\n"
))

cells.append(code(
    "SEED = 42\n"
    "\n"
    "def set_global_seeds(seed=SEED):\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    random.seed(seed)\n"
    "    np.random.seed(seed)\n"
    "    tf.random.set_seed(seed)\n"
    "    try:\n"
    "        tf.config.experimental.enable_op_determinism()\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "set_global_seeds(SEED)\n"
    "\n"
    "CONFIG = {\n"
    "    'DATA_PATH': r'D:\\PROJECTS\\NLP - PROJECT\\political_reddit_context_100k.csv',\n"
    "    'ARTIFACTS_DIR': 'artifacts',\n"
    "    'SEP_TOKEN': '[SEP]',\n"
    "    'SEP_PLACEHOLDER': 'xxsepxx',\n"
    "    'MAX_LEN': 200,\n"
    "    'EMBED_DIM': 100,\n"
    "    'W2V_WINDOW': 5,\n"
    "    'W2V_MIN_COUNT': 2,\n"
    "    'W2V_EPOCHS': 10,\n"
    "    'W2V_SG': 0,           # CBOW (sg=0)\n"
    "    'BATCH_SIZE': 64,\n"
    "    'EPOCHS': 30,\n"
    "    'LR': 0.001,\n"
    "    'CONV_FILTERS': 128,\n"
    "    'KERNEL_SIZE': 5,\n"
    "    'POOL_SIZE': 2,\n"
    "    'LSTM_UNITS': 128,\n"
    "    'DROPOUT': 0.3,\n"
    "    'SPATIAL_DROPOUT': 0.2,\n"
    "    'PATIENCE_ES': 4,\n"
    "    'PATIENCE_LR': 2,\n"
    "    'TEST_SIZE': 0.10,\n"
    "    'VAL_SIZE_OF_REMAINDER': 1.0 / 9.0,\n"
    "    'SEED': SEED,\n"
    "}\n"
    "\n"
    "Path(CONFIG['ARTIFACTS_DIR']).mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "print('TensorFlow:', tf.__version__)\n"
    "print('GPUs:', tf.config.list_physical_devices('GPU'))\n"
    "print('Seed:', SEED)\n"
    "print('Artifacts dir:', os.path.abspath(CONFIG['ARTIFACTS_DIR']))\n"
))

# ============================================================
# SECTION 2 — Data Loading
# ============================================================
cells.append(md("## 2. Data Loading\n"))

cells.append(code(
    "df = pd.read_csv(\n"
    "    CONFIG['DATA_PATH'],\n"
    "    dtype={'id': str, 'root': str, 'reply_to': str},\n"
    ")\n"
    "print('Shape:', df.shape)\n"
    "print('\\nDtypes:')\n"
    "print(df.dtypes)\n"
    "print('\\nFirst rows:')\n"
    "df.head()\n"
))

# ============================================================
# SECTION 3 — Integrity Checks
# ============================================================
cells.append(md("## 3. Dataset Integrity Checks\n"))

cells.append(code(
    "required = {'text', 'label', 'id', 'root', 'reply_to'}\n"
    "missing = required - set(df.columns)\n"
    "assert not missing, f'Missing columns: {missing}'\n"
    "\n"
    "df['label'] = df['label'].astype(int)\n"
    "print('Label values:', sorted(df['label'].unique()))\n"
    "assert set(df['label'].unique()) == {0, 1}\n"
    "\n"
    "print('\\nClass counts:')\n"
    "print(df['label'].value_counts())\n"
    "\n"
    "print('\\nNull counts per column:')\n"
    "print(df.isnull().sum())\n"
    "\n"
    "deleted_count = (df['text'] == '[deleted]').sum()\n"
    "removed_count = (df['text'] == '[removed]').sum()\n"
    "print(f'\\n[deleted] in text: {deleted_count}')\n"
    "print(f'[removed] in text: {removed_count}')\n"
    "\n"
    "dup_ids = df['id'].duplicated().sum()\n"
    "print(f'\\nDuplicate ids: {dup_ids}')\n"
    "\n"
    "df['char_len'] = df['text'].astype(str).str.len()\n"
    "print('\\nCharacter-length stats per class:')\n"
    "print(df.groupby('label')['char_len'].describe())\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "df['label'].value_counts().sort_index().plot(\n"
    "    kind='bar', ax=axes[0], color=['tab:blue', 'tab:red']\n"
    ")\n"
    "axes[0].set_xticklabels(['Democrat (0)', 'Republican (1)'], rotation=0)\n"
    "axes[0].set_title('Class distribution')\n"
    "axes[0].set_ylabel('Count')\n"
    "\n"
    "for lbl, color in [(0, 'tab:blue'), (1, 'tab:red')]:\n"
    "    axes[1].hist(\n"
    "        df.loc[df['label'] == lbl, 'char_len'].clip(upper=1000),\n"
    "        bins=60, alpha=0.5, label=f'label={lbl}', color=color,\n"
    "    )\n"
    "axes[1].set_title('Character length (clipped at 1000)')\n"
    "axes[1].legend()\n"
    "axes[1].set_xlabel('chars')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ============================================================
# SECTION 4 — Context Matching Logic
# ============================================================
cells.append(md("## 4. Context Matching Logic\n"))

cells.append(code(
    "def is_main_post(row):\n"
    "    return (row['id'] == row['root']) or pd.isna(row['reply_to'])\n"
    "\n"
    "def is_usable_text(t):\n"
    "    if not isinstance(t, str):\n"
    "        return False\n"
    "    s = t.strip()\n"
    "    if s == '' or s == '[deleted]' or s == '[removed]':\n"
    "        return False\n"
    "    return True\n"
    "\n"
    "main_post_mask = df.apply(is_main_post, axis=1) & df['text'].apply(is_usable_text)\n"
    "posts_df = df[main_post_mask]\n"
    "print(f'Total rows:           {len(df)}')\n"
    "print(f'Candidate main posts: {len(posts_df)}')\n"
    "\n"
    "posts_context = {}\n"
    "conflicts = 0\n"
    "for pid, txt in zip(posts_df['id'].tolist(), posts_df['text'].tolist()):\n"
    "    if pid in posts_context:\n"
    "        conflicts += 1\n"
    "        continue\n"
    "    posts_context[pid] = txt\n"
    "print(f'\\nposts_context size:   {len(posts_context)}')\n"
    "print(f'Conflicts skipped:    {conflicts}')\n"
    "\n"
    "with open(os.path.join(CONFIG['ARTIFACTS_DIR'], 'posts_context.pkl'), 'wb') as f:\n"
    "    pickle.dump(posts_context, f)\n"
    "\n"
    "rng = random.Random(SEED)\n"
    "if len(posts_context) >= 3:\n"
    "    sample_keys = rng.sample(list(posts_context.keys()), 3)\n"
    "    for k in sample_keys:\n"
    "        print(f'\\nid={k}')\n"
    "        print(f'text={posts_context[k][:200]!r}')\n"
))

cells.append(code(
    "# Context hit rate analysis (overall and per class)\n"
    "is_comment = df['id'] != df['root']\n"
    "df_comments = df[is_comment].copy()\n"
    "df_comments['context_found'] = df_comments['root'].isin(posts_context)\n"
    "\n"
    "overall_hit = df_comments['context_found'].mean()\n"
    "per_class_hit = df_comments.groupby('label')['context_found'].mean()\n"
    "\n"
    "print(f'Comments total:        {len(df_comments)}')\n"
    "print(f'Overall context hit:   {overall_hit:.4f}')\n"
    "print('\\nPer-class hit rate:')\n"
    "print(per_class_hit)\n"
))

# ============================================================
# SECTION 5 — Context Enrichment
# ============================================================
cells.append(md("## 5. Context Enrichment\n"))

cells.append(code(
    "SEP_PLACEHOLDER = CONFIG['SEP_PLACEHOLDER']\n"
    "\n"
    "def build_enriched(row):\n"
    "    txt = row['text'] if isinstance(row['text'], str) else ''\n"
    "    if row['id'] == row['root']:\n"
    "        return txt, False  # the row itself is a top-level post\n"
    "    parent = posts_context.get(row['root'])\n"
    "    if parent is None:\n"
    "        return txt, False  # fallback: comment-only\n"
    "    return parent + ' ' + SEP_PLACEHOLDER + ' ' + txt, True\n"
    "\n"
    "results = df.apply(build_enriched, axis=1)\n"
    "df['enriched_text'] = [r[0] for r in results]\n"
    "df['has_context']  = [r[1] for r in results]\n"
    "\n"
    "print('has_context distribution:')\n"
    "print(df['has_context'].value_counts())\n"
    "print('\\nMean has_context per label:')\n"
    "print(df.groupby('label')['has_context'].mean())\n"
    "\n"
    "print('\\n=== Sample enriched rows (has_context=True) ===')\n"
    "ex = df[df['has_context']].sample(3, random_state=SEED)\n"
    "for _, row in ex.iterrows():\n"
    "    print('\\n---')\n"
    "    print('LABEL   :', row['label'])\n"
    "    print('ENRICHED:', row['enriched_text'][:400])\n"
    "\n"
    "print('\\n=== Sample fallback rows (has_context=False, not a post) ===')\n"
    "fallback = df[(~df['has_context']) & (df['id'] != df['root'])]\n"
    "if len(fallback) > 0:\n"
    "    for _, row in fallback.sample(min(3, len(fallback)), random_state=SEED).iterrows():\n"
    "        print('\\n---')\n"
    "        print('LABEL   :', row['label'])\n"
    "        print('TEXT    :', row['enriched_text'][:400])\n"
))

# ============================================================
# SECTION 6 — Preprocessing
# ============================================================
cells.append(md("## 6. Preprocessing Pipeline\n"
                "\n"
                "Order: tokenize → lowercase → stopword removal → stemming. "
                "The `[SEP]` marker is replaced with a stem-proof placeholder (`xxsepxx`) before "
                "tokenization so the stemmer and stopword filter cannot mutate or delete it.\n"))

cells.append(code(
    "stemmer = PorterStemmer()\n"
    "stop_words = set(stopwords.words('english'))\n"
    "PLACEHOLDER = CONFIG['SEP_PLACEHOLDER']\n"
    "SEP_TOKEN = CONFIG['SEP_TOKEN']\n"
    "\n"
    "def preprocess(text):\n"
    "    if not isinstance(text, str):\n"
    "        return []\n"
    "    # 1) Protect the SEP marker\n"
    "    text = text.replace(SEP_TOKEN, ' ' + PLACEHOLDER + ' ')\n"
    "    # 2) Tokenize\n"
    "    tokens = word_tokenize(text)\n"
    "    # 3) Lowercase\n"
    "    tokens = [t.lower() for t in tokens]\n"
    "    # 4) Stopword removal (preserve placeholder)\n"
    "    tokens = [t for t in tokens if (t == PLACEHOLDER) or (t not in stop_words)]\n"
    "    # 5) Stemming (preserve placeholder)\n"
    "    tokens = [t if t == PLACEHOLDER else stemmer.stem(t) for t in tokens]\n"
    "    return tokens\n"
))

cells.append(code(
    "try:\n"
    "    from tqdm.auto import tqdm\n"
    "    tqdm.pandas(desc='Preprocessing')\n"
    "    df['tokens'] = df['enriched_text'].progress_apply(preprocess)\n"
    "except Exception:\n"
    "    df['tokens'] = df['enriched_text'].apply(preprocess)\n"
    "\n"
    "df['n_tokens'] = df['tokens'].apply(len)\n"
    "\n"
    "print('Token-length stats:')\n"
    "print(df['n_tokens'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))\n"
    "\n"
    "n_empty = int((df['n_tokens'] == 0).sum())\n"
    "print(f'\\nEmpty-token rows: {n_empty}')\n"
    "if n_empty > 0:\n"
    "    df = df[df['n_tokens'] > 0].reset_index(drop=True)\n"
    "    print(f'After dropping empties: {len(df)}')\n"
    "\n"
    "# Placeholder integrity check\n"
    "has_ctx = df[df['has_context']]\n"
    "contains_ph = has_ctx['tokens'].apply(lambda toks: PLACEHOLDER in toks)\n"
    "print(f'\\nhas_context rows:                                  {len(has_ctx)}')\n"
    "print(f'has_context rows that still contain the placeholder: {int(contains_ph.sum())}')\n"
    "\n"
    "print('\\nSample tokenized rows:')\n"
    "for i in range(3):\n"
    "    print(f'\\nrow {i} label={df[\"label\"].iloc[i]}')\n"
    "    print(df['tokens'].iloc[i][:40])\n"
))

# ============================================================
# SECTION 7 — Stratified Split
# ============================================================
cells.append(md("## 7. Stratified Train / Validation / Test Split (80 / 10 / 10)\n"))

cells.append(code(
    "train_val_df, test_df = train_test_split(\n"
    "    df,\n"
    "    test_size=CONFIG['TEST_SIZE'],\n"
    "    stratify=df['label'],\n"
    "    random_state=SEED,\n"
    ")\n"
    "train_df, val_df = train_test_split(\n"
    "    train_val_df,\n"
    "    test_size=CONFIG['VAL_SIZE_OF_REMAINDER'],\n"
    "    stratify=train_val_df['label'],\n"
    "    random_state=SEED,\n"
    ")\n"
    "train_df = train_df.reset_index(drop=True)\n"
    "val_df   = val_df.reset_index(drop=True)\n"
    "test_df  = test_df.reset_index(drop=True)\n"
    "\n"
    "n_total = len(train_df) + len(val_df) + len(test_df)\n"
    "print(f'Train: {len(train_df):>6} ({len(train_df)/n_total:.2%})')\n"
    "print(f'Val:   {len(val_df):>6} ({len(val_df)/n_total:.2%})')\n"
    "print(f'Test:  {len(test_df):>6} ({len(test_df)/n_total:.2%})')\n"
    "\n"
    "print('\\nLabel balance per split:')\n"
    "for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:\n"
    "    print(f'  {name}: {split[\"label\"].value_counts(normalize=True).round(4).to_dict()}')\n"
    "\n"
    "print('\\nhas_context fraction per split:')\n"
    "for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:\n"
    "    print(f'  {name}: {split[\"has_context\"].mean():.4f}')\n"
))

# ============================================================
# SECTION 8 — Token length analysis
# ============================================================
cells.append(md("## 8. Token-Length Analysis (Train Only) → choose `MAX_LEN`\n"))

cells.append(code(
    "train_lens = train_df['n_tokens']\n"
    "print('Train token-length stats:')\n"
    "print(train_lens.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))\n"
    "\n"
    "p95 = int(np.percentile(train_lens, 95))\n"
    "p99 = int(np.percentile(train_lens, 99))\n"
    "print(f'\\nP95: {p95}   P99: {p99}')\n"
    "\n"
    "MAX_LEN = int(min(max(p95, 64), 300))\n"
    "CONFIG['MAX_LEN'] = MAX_LEN\n"
    "print(f'\\nUsing MAX_LEN = {MAX_LEN}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
    "axes[0].hist(train_lens, bins=80)\n"
    "axes[0].axvline(MAX_LEN, color='red', linestyle='--', label=f'MAX_LEN={MAX_LEN}')\n"
    "axes[0].set_title('Train token length (full)')\n"
    "axes[0].set_xlabel('# tokens')\n"
    "axes[0].legend()\n"
    "\n"
    "clipped = train_lens.clip(upper=MAX_LEN * 2)\n"
    "axes[1].hist(clipped, bins=80)\n"
    "axes[1].axvline(MAX_LEN, color='red', linestyle='--', label=f'MAX_LEN={MAX_LEN}')\n"
    "axes[1].set_title('Train token length (clipped view)')\n"
    "axes[1].set_xlabel('# tokens')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ============================================================
# SECTION 9 — Tokenizer + sequences
# ============================================================
cells.append(md("## 9. Keras Tokenizer and Padded Sequences\n"
                "\n"
                "Tokenizer is fit on **train only**. `filters=''` and `lower=False` prevent it from "
                "re-stripping the `[SEP]` placeholder or undoing case decisions already made.\n"))

cells.append(code(
    "train_token_strings = train_df['tokens'].apply(lambda toks: ' '.join(toks)).tolist()\n"
    "val_token_strings   = val_df['tokens'].apply(lambda toks: ' '.join(toks)).tolist()\n"
    "test_token_strings  = test_df['tokens'].apply(lambda toks: ' '.join(toks)).tolist()\n"
    "\n"
    "tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')\n"
    "tokenizer.fit_on_texts(train_token_strings)\n"
    "\n"
    "vocab_size = len(tokenizer.word_index) + 1\n"
    "print(f'Vocab size (incl. pad+OOV): {vocab_size}')\n"
    "\n"
    "X_train_seq = tokenizer.texts_to_sequences(train_token_strings)\n"
    "X_val_seq   = tokenizer.texts_to_sequences(val_token_strings)\n"
    "X_test_seq  = tokenizer.texts_to_sequences(test_token_strings)\n"
    "\n"
    "X_train = pad_sequences(X_train_seq, maxlen=CONFIG['MAX_LEN'], padding='post', truncating='post')\n"
    "X_val   = pad_sequences(X_val_seq,   maxlen=CONFIG['MAX_LEN'], padding='post', truncating='post')\n"
    "X_test  = pad_sequences(X_test_seq,  maxlen=CONFIG['MAX_LEN'], padding='post', truncating='post')\n"
    "\n"
    "y_train = train_df['label'].values.astype('float32')\n"
    "y_val   = val_df['label'].values.astype('float32')\n"
    "y_test  = test_df['label'].values.astype('float32')\n"
    "\n"
    "print(f'X_train: {X_train.shape}  y_train: {y_train.shape}')\n"
    "print(f'X_val:   {X_val.shape}    y_val:   {y_val.shape}')\n"
    "print(f'X_test:  {X_test.shape}   y_test:  {y_test.shape}')\n"
    "\n"
    "oov_idx = tokenizer.word_index.get('<OOV>')\n"
    "def oov_fraction(seqs, oov_id):\n"
    "    total = sum(len(s) for s in seqs)\n"
    "    if total == 0 or oov_id is None:\n"
    "        return 0.0\n"
    "    oov = sum(1 for s in seqs for i in s if i == oov_id)\n"
    "    return oov / total\n"
    "\n"
    "print(f'\\nOOV fraction (train): {oov_fraction(X_train_seq, oov_idx):.4f}')\n"
    "print(f'OOV fraction (val):   {oov_fraction(X_val_seq, oov_idx):.4f}')\n"
    "print(f'OOV fraction (test):  {oov_fraction(X_test_seq, oov_idx):.4f}')\n"
    "\n"
    "with open(os.path.join(CONFIG['ARTIFACTS_DIR'], 'tokenizer.pkl'), 'wb') as f:\n"
    "    pickle.dump(tokenizer, f)\n"
))

# ============================================================
# SECTION 10 — Word2Vec CBOW
# ============================================================
cells.append(md("## 10. CBOW Word2Vec Training (Train Corpus Only)\n"))

cells.append(code(
    "train_tokens_list = train_df['tokens'].tolist()\n"
    "print(f'Sentences for Word2Vec: {len(train_tokens_list)}')\n"
    "\n"
    "w2v = Word2Vec(\n"
    "    sentences=train_tokens_list,\n"
    "    vector_size=CONFIG['EMBED_DIM'],\n"
    "    window=CONFIG['W2V_WINDOW'],\n"
    "    min_count=CONFIG['W2V_MIN_COUNT'],\n"
    "    workers=4,\n"
    "    sg=CONFIG['W2V_SG'],   # 0 = CBOW (required)\n"
    "    epochs=CONFIG['W2V_EPOCHS'],\n"
    "    seed=SEED,\n"
    ")\n"
    "assert CONFIG['W2V_SG'] == 0, 'CBOW is mandatory'\n"
    "\n"
    "print(f'W2V vocab size: {len(w2v.wv)}')\n"
    "w2v.save(os.path.join(CONFIG['ARTIFACTS_DIR'], 'w2v_cbow.model'))\n"
    "\n"
    "probe_words = ['trump', 'biden', 'tax', 'gun', 'liber', 'conserv', PLACEHOLDER]\n"
    "for w in probe_words:\n"
    "    if w in w2v.wv:\n"
    "        print(f\"\\nNearest to '{w}':\")\n"
    "        for sim_word, score in w2v.wv.most_similar(w, topn=5):\n"
    "            print(f'  {sim_word:>20s}  {score:.3f}')\n"
    "    else:\n"
    "        print(f\"\\n'{w}' not in W2V vocab\")\n"
))

# ============================================================
# SECTION 11 — Embedding Matrix
# ============================================================
cells.append(md("## 11. Embedding Matrix Construction\n"))

cells.append(code(
    "embedding_matrix = np.zeros((vocab_size, CONFIG['EMBED_DIM']), dtype='float32')\n"
    "hits = 0\n"
    "for word, idx in tokenizer.word_index.items():\n"
    "    if word in w2v.wv:\n"
    "        embedding_matrix[idx] = w2v.wv[word]\n"
    "        hits += 1\n"
    "\n"
    "embedding_matrix[0] = 0.0  # padding row stays zero\n"
    "\n"
    "coverage = hits / max(vocab_size - 1, 1)\n"
    "print(f'Embedding matrix shape: {embedding_matrix.shape}')\n"
    "print(f'W2V coverage:           {coverage:.4f} ({hits}/{vocab_size - 1})')\n"
    "\n"
    "np.save(os.path.join(CONFIG['ARTIFACTS_DIR'], 'embedding_matrix.npy'), embedding_matrix)\n"
    "\n"
    "assert embedding_matrix.shape == (vocab_size, CONFIG['EMBED_DIM'])\n"
    "assert np.all(embedding_matrix[0] == 0.0)\n"
))

# ============================================================
# SECTION 12 — CNN-LSTM Model
# ============================================================
cells.append(md("## 12. CNN-LSTM Model Definition\n"
                "\n"
                "Architecture: `Embedding (W2V init, trainable) → SpatialDropout1D → Conv1D → "
                "MaxPooling1D → LSTM → Dropout → Dense(sigmoid)`.\n"))

cells.append(code(
    "def build_cnn_lstm(vocab_size, embed_dim, max_len, embedding_matrix, config):\n"
    "    inp = Input(shape=(max_len,), name='input_ids')\n"
    "    x = Embedding(\n"
    "        input_dim=vocab_size,\n"
    "        output_dim=embed_dim,\n"
    "        weights=[embedding_matrix],\n"
    "        input_length=max_len,\n"
    "        trainable=True,\n"
    "        mask_zero=False,\n"
    "        name='embedding',\n"
    "    )(inp)\n"
    "    x = SpatialDropout1D(config['SPATIAL_DROPOUT'])(x)\n"
    "    x = Conv1D(\n"
    "        filters=config['CONV_FILTERS'],\n"
    "        kernel_size=config['KERNEL_SIZE'],\n"
    "        padding='same',\n"
    "        activation='relu',\n"
    "        name='conv1d',\n"
    "    )(x)\n"
    "    x = MaxPooling1D(pool_size=config['POOL_SIZE'], name='maxpool')(x)\n"
    "    x = LSTM(\n"
    "        config['LSTM_UNITS'],\n"
    "        dropout=0.2,\n"
    "        recurrent_dropout=0.0,\n"
    "        name='lstm',\n"
    "    )(x)\n"
    "    x = Dropout(config['DROPOUT'])(x)\n"
    "    out = Dense(1, activation='sigmoid', name='output')(x)\n"
    "    return Model(inp, out, name='cnn_lstm')\n"
    "\n"
    "model = build_cnn_lstm(\n"
    "    vocab_size=vocab_size,\n"
    "    embed_dim=CONFIG['EMBED_DIM'],\n"
    "    max_len=CONFIG['MAX_LEN'],\n"
    "    embedding_matrix=embedding_matrix,\n"
    "    config=CONFIG,\n"
    ")\n"
    "model.compile(\n"
    "    optimizer=Adam(learning_rate=CONFIG['LR']),\n"
    "    loss='binary_crossentropy',\n"
    "    metrics=['accuracy'],\n"
    ")\n"
    "model.summary()\n"
))

# ============================================================
# SECTION 13 — Callbacks
# ============================================================
cells.append(md("## 13. Training Callbacks\n"))

cells.append(code(
    "best_path = os.path.join(CONFIG['ARTIFACTS_DIR'], 'best_model.keras')\n"
    "\n"
    "callbacks = [\n"
    "    ModelCheckpoint(\n"
    "        filepath=best_path,\n"
    "        monitor='val_accuracy',\n"
    "        mode='max',\n"
    "        save_best_only=True,\n"
    "        verbose=1,\n"
    "    ),\n"
    "    EarlyStopping(\n"
    "        monitor='val_accuracy',\n"
    "        mode='max',\n"
    "        patience=CONFIG['PATIENCE_ES'],\n"
    "        restore_best_weights=True,\n"
    "        verbose=1,\n"
    "    ),\n"
    "    ReduceLROnPlateau(\n"
    "        monitor='val_loss',\n"
    "        factor=0.5,\n"
    "        patience=CONFIG['PATIENCE_LR'],\n"
    "        min_lr=1e-5,\n"
    "        verbose=1,\n"
    "    ),\n"
    "]\n"
    "print(f'Best model will be saved to: {best_path}')\n"
))

# ============================================================
# SECTION 14 — Training
# ============================================================
cells.append(md("## 14. Model Training\n"))

cells.append(code(
    "history = model.fit(\n"
    "    X_train, y_train,\n"
    "    validation_data=(X_val, y_val),\n"
    "    epochs=CONFIG['EPOCHS'],\n"
    "    batch_size=CONFIG['BATCH_SIZE'],\n"
    "    callbacks=callbacks,\n"
    "    shuffle=True,\n"
    "    verbose=2,\n"
    ")\n"
    "\n"
    "hist_path = os.path.join(CONFIG['ARTIFACTS_DIR'], 'history.json')\n"
    "with open(hist_path, 'w') as f:\n"
    "    json.dump(\n"
    "        {k: [float(x) for x in v] for k, v in history.history.items()},\n"
    "        f, indent=2,\n"
    "    )\n"
    "\n"
    "print(f'\\nTrained for {len(history.history[\"loss\"])} epochs')\n"
    "print(f'Best val_accuracy: {max(history.history[\"val_accuracy\"]):.4f}')\n"
))

# ============================================================
# SECTION 15 — Validation Curves
# ============================================================
cells.append(md("## 15. Training & Validation Curves\n"))

cells.append(code(
    "hist = history.history\n"
    "best_ep = int(np.argmax(hist['val_accuracy']))\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "axes[0].plot(hist['accuracy'], label='Train Accuracy')\n"
    "axes[0].plot(hist['val_accuracy'], label='Val Accuracy')\n"
    "axes[0].axvline(best_ep, color='red', linestyle='--', label=f'Best epoch ({best_ep})')\n"
    "axes[0].set_title('Accuracy')\n"
    "axes[0].set_xlabel('Epoch')\n"
    "axes[0].set_ylabel('Accuracy')\n"
    "axes[0].legend()\n"
    "axes[0].grid(alpha=0.3)\n"
    "\n"
    "axes[1].plot(hist['loss'], label='Train Loss')\n"
    "axes[1].plot(hist['val_loss'], label='Val Loss')\n"
    "axes[1].axvline(best_ep, color='red', linestyle='--', label=f'Best epoch ({best_ep})')\n"
    "axes[1].set_title('Loss')\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].set_ylabel('Loss')\n"
    "axes[1].legend()\n"
    "axes[1].grid(alpha=0.3)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ============================================================
# SECTION 16 — Final Test Evaluation
# ============================================================
cells.append(md("## 16. Final Test Evaluation\n"))

cells.append(code(
    "best_model = load_model(best_path)\n"
    "\n"
    "y_prob = best_model.predict(X_test, batch_size=CONFIG['BATCH_SIZE'], verbose=1).ravel()\n"
    "y_pred = (y_prob >= 0.5).astype(int)\n"
    "y_true = y_test.astype(int)\n"
    "\n"
    "acc        = accuracy_score(y_true, y_pred)\n"
    "prec_rep   = precision_score(y_true, y_pred, pos_label=1)\n"
    "rec_rep    = recall_score(y_true, y_pred, pos_label=1)\n"
    "f1_rep     = f1_score(y_true, y_pred, pos_label=1)\n"
    "prec_macro = precision_score(y_true, y_pred, average='macro')\n"
    "rec_macro  = recall_score(y_true, y_pred, average='macro')\n"
    "f1_macro   = f1_score(y_true, y_pred, average='macro')\n"
    "\n"
    "print(f'Test Accuracy:           {acc:.4f}')\n"
    "print(f'Test Precision (Rep):    {prec_rep:.4f}')\n"
    "print(f'Test Recall    (Rep):    {rec_rep:.4f}')\n"
    "print(f'Test F1        (Rep):    {f1_rep:.4f}')\n"
    "print(f'Test Precision (macro):  {prec_macro:.4f}')\n"
    "print(f'Test Recall    (macro):  {rec_macro:.4f}')\n"
    "print(f'Test F1        (macro):  {f1_macro:.4f}')\n"
    "\n"
    "print('\\nClassification report:')\n"
    "print(classification_report(\n"
    "    y_true, y_pred,\n"
    "    target_names=['Democrat', 'Republican'],\n"
    "    digits=4,\n"
    "))\n"
    "\n"
    "results = {\n"
    "    'accuracy': float(acc),\n"
    "    'precision_republican': float(prec_rep),\n"
    "    'recall_republican':    float(rec_rep),\n"
    "    'f1_republican':        float(f1_rep),\n"
    "    'precision_macro':      float(prec_macro),\n"
    "    'recall_macro':         float(rec_macro),\n"
    "    'f1_macro':             float(f1_macro),\n"
    "    'n_test':               int(len(y_true)),\n"
    "    'max_len':              int(CONFIG['MAX_LEN']),\n"
    "    'vocab_size':           int(vocab_size),\n"
    "}\n"
    "with open(os.path.join(CONFIG['ARTIFACTS_DIR'], 'test_metrics.json'), 'w') as f:\n"
    "    json.dump(results, f, indent=2)\n"
))

# ============================================================
# SECTION 17 — Confusion Matrix
# ============================================================
cells.append(md("## 17. Confusion Matrix Visualization\n"))

cells.append(code(
    "cm = confusion_matrix(y_true, y_pred)\n"
    "print('Confusion matrix (rows = true, cols = predicted):')\n"
    "print(cm)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "sns.heatmap(\n"
    "    cm, annot=True, fmt='d', cmap='Blues',\n"
    "    xticklabels=['Democrat', 'Republican'],\n"
    "    yticklabels=['Democrat', 'Republican'],\n"
    "    ax=ax, cbar=False,\n"
    ")\n"
    "ax.set_xlabel('Predicted')\n"
    "ax.set_ylabel('True')\n"
    "ax.set_title(f'Confusion Matrix (Test) — Acc {acc:.4f}')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ============================================================
# SECTION 18 — Error Analysis / Sample Predictions
# ============================================================
cells.append(md("## 18. Error Analysis / Sample Predictions\n"))

cells.append(code(
    "analysis_df = test_df.copy().reset_index(drop=True)\n"
    "analysis_df['y_true'] = y_true\n"
    "analysis_df['y_pred'] = y_pred\n"
    "analysis_df['y_prob'] = y_prob\n"
    "analysis_df['correct'] = (analysis_df['y_true'] == analysis_df['y_pred'])\n"
    "analysis_df['margin']  = (analysis_df['y_prob'] - 0.5).abs()\n"
    "\n"
    "print(f'Total test rows:   {len(analysis_df)}')\n"
    "print(f'Correct:           {int(analysis_df[\"correct\"].sum())} '\n"
    "      f'({analysis_df[\"correct\"].mean():.4f})')\n"
    "print(f'has_context frac:  {analysis_df[\"has_context\"].mean():.4f}')\n"
    "\n"
    "print('\\nAccuracy by has_context:')\n"
    "print(analysis_df.groupby('has_context')['correct'].agg(['mean', 'count']))\n"
    "\n"
    "print('\\nAccuracy by true label:')\n"
    "print(analysis_df.groupby('y_true')['correct'].agg(['mean', 'count']))\n"
))

cells.append(code(
    "def show(rows, title):\n"
    "    print('\\n' + '=' * 78)\n"
    "    print(title)\n"
    "    print('=' * 78)\n"
    "    for _, r in rows.iterrows():\n"
    "        print(\n"
    "            f\"\\n[true={int(r['y_true'])} pred={int(r['y_pred'])} \"\n"
    "            f\"prob={r['y_prob']:.3f} has_context={bool(r['has_context'])}]\"\n"
    "        )\n"
    "        text = r['enriched_text']\n"
    "        print(text[:400] + ('...' if len(text) > 400 else ''))\n"
    "\n"
    "top_correct_rep = analysis_df[(analysis_df['correct']) & (analysis_df['y_true'] == 1)].nlargest(5, 'y_prob')\n"
    "top_correct_dem = analysis_df[(analysis_df['correct']) & (analysis_df['y_true'] == 0)].nsmallest(5, 'y_prob')\n"
    "top_wrong_rep   = analysis_df[(~analysis_df['correct']) & (analysis_df['y_true'] == 1)].nsmallest(5, 'y_prob')\n"
    "top_wrong_dem   = analysis_df[(~analysis_df['correct']) & (analysis_df['y_true'] == 0)].nlargest(5, 'y_prob')\n"
    "ambiguous       = analysis_df.nsmallest(5, 'margin')\n"
    "\n"
    "show(top_correct_rep, 'Top confident CORRECT — Republican')\n"
    "show(top_correct_dem, 'Top confident CORRECT — Democrat')\n"
    "show(top_wrong_rep,   'Top confident WRONG  — Republican (predicted Democrat)')\n"
    "show(top_wrong_dem,   'Top confident WRONG  — Democrat (predicted Republican)')\n"
    "show(ambiguous,       'Most ambiguous predictions (prob near 0.5)')\n"
    "\n"
    "err_path = os.path.join(CONFIG['ARTIFACTS_DIR'], 'error_analysis.csv')\n"
    "analysis_df[[\n"
    "    'text', 'enriched_text', 'has_context',\n"
    "    'y_true', 'y_pred', 'y_prob', 'correct', 'margin',\n"
    "]].to_csv(err_path, index=False)\n"
    "print(f'\\nFull error analysis saved to: {err_path}')\n"
))

# ============================================================
# Closing
# ============================================================
cells.append(md(
    "## Done\n"
    "\n"
    "Artifacts saved under `artifacts/`:\n"
    "\n"
    "- `posts_context.pkl` — id → post text dictionary\n"
    "- `tokenizer.pkl` — Keras tokenizer fit on train\n"
    "- `w2v_cbow.model` — domain-specific CBOW Word2Vec\n"
    "- `embedding_matrix.npy` — aligned embedding matrix\n"
    "- `best_model.keras` — best validation checkpoint\n"
    "- `history.json` — training curves\n"
    "- `test_metrics.json` — final test metrics\n"
    "- `error_analysis.csv` — per-row test predictions for inspection\n"
))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells.")

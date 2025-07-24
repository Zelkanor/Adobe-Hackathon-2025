# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# from scipy.sparse import hstack, csr_matrix
# import joblib
# import os
# import warnings
# from pathlib import Path

# warnings.filterwarnings('ignore')
# logger = print

# # --- Path Configuration ---
# TRAINING_DIR = Path(__file__).resolve().parent
# ROOT_DIR = TRAINING_DIR.parent
# DATA_DIR = ROOT_DIR / 'data'

# # --- Main Configuration ---
# SOURCE_CSV_FILE = DATA_DIR / 'merged_features2.csv'
# CLEANED_FOR_TRAINING_CSV = DATA_DIR / 'cleaned_for_training.csv'
# MODEL_DIR = ROOT_DIR / 'final_heading_model'

# TARGET_LABELS = ['H1', 'H2', 'H3']
# CHUNK_SIZE = 100_000


# def clean_data_for_training(source_path: Path, output_path: Path):
#     """
#     Loads the source CSV, robustly cleans all data types, uses heuristics
#     to remove titles, and saves a file ready for model training.
#     """
#     logger(f"🔄 Preparing and cleaning {source_path} for training...")
#     if not source_path.exists():
#         raise FileNotFoundError(f"Source file not found: {source_path}")

#     df = pd.read_csv(source_path, low_memory=False)

#     # --- Robust Data Type Conversion (CORRECTED) ---
#     numerical_features = [
#         'font_size', 'font_size_rank', 'relative_font_size', 'caps_ratio',
#         'word_count', 'title_case_ratio', 'punctuation', 'whitespace_above',
#         'whitespace_below', 'bbox_width', 'bbox_height', 'relative_position',
#         'page_number'
#     ]
#     bool_features = ['bold', 'italic', 'underline', 'has_numbering']
    
#     logger("   Converting data types for training...")
#     for col in numerical_features:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#     for col in bool_features:
#         if col in df.columns:
#             # Convert boolean-like columns (True, "True", 1, etc.) to a consistent bool type first
#             df[col] = df[col].apply(lambda x: str(x).lower() in ['true', '1', '1.0'])
    
#     # Now, convert all boolean columns to integers (1/0)
#     if all(col in df.columns for col in bool_features):
#         df[bool_features] = df[bool_features].astype(int)
            
#     # After all conversions, we can safely fill any missing numbers
#     df[numerical_features] = df[numerical_features].fillna(0)
    
#     # --- Heuristic to Find and Remove Titles ---
#     def find_title_index_in_group(df_group: pd.DataFrame) -> pd.Index:
#         """Finds the index of the most likely title in a document group."""
#         if df_group.empty: return pd.Index([])
#         first_page_df = df_group[df_group['page_number'] == df_group['page_number'].min()]
#         if first_page_df.empty: return pd.Index([])
        
#         max_font = first_page_df['font_size'].max()
#         if max_font == 0: return pd.Index([])
        
#         candidates = first_page_df[first_page_df['font_size'] >= max_font * 0.9]
#         if candidates.empty: return pd.Index([])
#         return candidates.nsmallest(1, 'relative_position').index

#     logger("   Applying heuristics to identify and exclude titles...")
#     title_indices = df.groupby('FileID', group_keys=False).apply(find_title_index_in_group)
    
#     valid_title_indices = title_indices[title_indices.apply(len) > 0]
#     if not valid_title_indices.empty:
#         title_indices_flat = np.concatenate(valid_title_indices.values)
#         df_no_titles = df.drop(index=title_indices_flat)
#     else:
#         df_no_titles = df

#     # --- Final Data Cleaning ---
#     target_col = 'heading_level' if 'heading_level' in df_no_titles.columns and df_no_titles['heading_level'].notna().any() else 'label'

#     final_df = df_no_titles[df_no_titles[target_col].isin(TARGET_LABELS)].copy()
#     final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

#     final_df.to_csv(output_path, index=False)
#     logger(f"✅ Cleaned data with {len(final_df)} heading samples saved to '{output_path}'")
#     return final_df, target_col


# def train_final_model(data_path: Path, target_col: str):
#     """Full training pipeline using the cleaned, unified dataset."""
#     logger("\n--- Starting Model Training ---")

#     logger("   Fitting TF-IDF vectorizer...")
#     vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
#     text_iterator = pd.read_csv(data_path, chunksize=CHUNK_SIZE, usecols=['text'], iterator=True)
#     vectorizer.fit(pd.concat(text_iterator)['text'].astype(str))

#     numerical_features = [
#         'font_size', 'font_size_rank', 'relative_font_size', 'bold', 'italic', 'underline',
#         'caps_ratio', 'word_count', 'title_case_ratio', 'has_numbering', 'punctuation',
#         'whitespace_above', 'whitespace_below', 'bbox_width', 'bbox_height', 'relative_position'
#     ]
#     categorical_features = ['alignment']
#     label_map = {label: i for i, label in enumerate(TARGET_LABELS)}

#     logger("   Training LightGBM model on data chunks...")
#     model = None
    
#     # --- MEMORY-SAFE TRAINING LOOP (CORRECTED) ---
#     # Use an iterator directly instead of converting to a list
#     chunk_iterator = pd.read_csv(data_path, chunksize=CHUNK_SIZE, iterator=True)
    
#     # Buffer to hold the last chunk for validation
#     validation_chunk = None
#     chunk_count = 0

#     for chunk in chunk_iterator:
#         if validation_chunk is not None:
#             # If we have a buffered chunk, train on it
#             y_train = validation_chunk[target_col].map(label_map)
            
#             # The data types are now guaranteed to be correct from the cleaning step
#             num_feat = csr_matrix(validation_chunk[numerical_features].values)
#             cat_feat = pd.get_dummies(validation_chunk[categorical_features], sparse=True, dtype=float).sparse.to_coo()
#             text_feat = vectorizer.transform(validation_chunk['text'].astype(str))
            
#             X_train = hstack([num_feat, cat_feat, text_feat], format='csr')

#             lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
#             model = lgb.train({'objective': 'multiclass', 'num_class': 3, 'verbose': -1},
#                               lgb_train, num_boost_round=15, init_model=model, keep_training_booster=True)
#             logger(f"   Trained on chunk {chunk_count}...")
        
#         # Move the current chunk to the buffer for the next iteration
#         validation_chunk = chunk
#         chunk_count += 1

#     if model is None:
#         logger("⚠️ Not enough data to train. Only one chunk found, which is reserved for validation. Please use a larger dataset or smaller CHUNK_SIZE.")
#         return

#     # --- Evaluate on the final held-out chunk ---
#     logger("\n📊 Evaluating model on final chunk...")
#     y_val = validation_chunk[target_col].map(label_map).dropna()
#     valid_indices = y_val.index
    
#     num_feat_val = csr_matrix(validation_chunk.loc[valid_indices, numerical_features].values)
#     cat_feat_val = pd.get_dummies(validation_chunk.loc[valid_indices, categorical_features], sparse=True, dtype=float).sparse.to_coo()
#     text_feat_val = vectorizer.transform(validation_chunk.loc[valid_indices, 'text'].astype(str))
#     X_val = hstack([num_feat_val, cat_feat_val, text_feat_val], format='csr')

#     y_pred = np.argmax(model.predict(X_val), axis=1)
#     print(classification_report(y_val, y_pred, target_names=TARGET_LABELS))

#     # --- Save All Artifacts ---
#     logger(f"💾 Saving artifacts to '{MODEL_DIR}/'...")
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     model.save_model(str(MODEL_DIR / 'lgbm_model.txt'))
#     joblib.dump(vectorizer, MODEL_DIR / 'tfidf_vectorizer.joblib')
#     joblib.dump(numerical_features, MODEL_DIR / 'numerical_features.joblib')
#     joblib.dump(categorical_features, MODEL_DIR / 'categorical_features.joblib')
#     joblib.dump(label_map, MODEL_DIR / 'label_map.joblib')
#     logger("✅ Training complete and all artifacts saved.")

# if __name__ == '__main__':
#     _, target_column = clean_data_for_training(SOURCE_CSV_FILE, CLEANED_FOR_TRAINING_CSV)
#     train_final_model(CLEANED_FOR_TRAINING_CSV, target_column)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import joblib
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
logger = print

# --- Path Configuration ---
TRAINING_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRAINING_DIR.parent
DATA_DIR = ROOT_DIR / 'data'

# --- Main Configuration ---
SOURCE_CSV_FILE = DATA_DIR / 'merged_features2.csv'
CLEANED_FOR_TRAINING_CSV = DATA_DIR / 'cleaned_for_training.csv'
MODEL_DIR = ROOT_DIR / 'final_heading_model'

TARGET_LABELS = ['H1', 'H2', 'H3']
CHUNK_SIZE = 100_000

def clean_data_for_training(source_path: Path, output_path: Path):
    """
    Loads the source CSV, robustly cleans all data types, uses heuristics
    to remove titles, and saves a file ready for model training.
    """
    logger(f"🔄 Preparing and cleaning {source_path} for training...")
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path, low_memory=False)

    numerical_features = [
        'font_size', 'font_size_rank', 'relative_font_size', 'caps_ratio',
        'word_count', 'title_case_ratio', 'punctuation', 'whitespace_above',
        'whitespace_below', 'bbox_width', 'bbox_height', 'relative_position',
        'page_number'
    ]
    bool_features = ['bold', 'italic', 'underline', 'has_numbering']
    
    logger("   Converting data types for training...")
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in bool_features:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).lower() in ['true', '1', '1.0'])
    
    if all(col in df.columns for col in bool_features):
        df[bool_features] = df[bool_features].astype(int)
            
    df[numerical_features] = df[numerical_features].fillna(0)
    
    def find_title_index_in_group(df_group: pd.DataFrame) -> pd.Index:
        if df_group.empty: return pd.Index([])
        first_page_df = df_group[df_group['page_number'] == df_group['page_number'].min()]
        if first_page_df.empty: return pd.Index([])
        max_font = first_page_df['font_size'].max()
        if max_font == 0: return pd.Index([])
        candidates = first_page_df[first_page_df['font_size'] >= max_font * 0.9]
        if candidates.empty: return pd.Index([])
        return candidates.nsmallest(1, 'relative_position').index

    logger("   Applying heuristics to identify and exclude titles...")
    title_indices = df.groupby('FileID', group_keys=False).apply(find_title_index_in_group)
    
    valid_title_indices = title_indices[title_indices.apply(len) > 0]
    if not valid_title_indices.empty:
        title_indices_flat = np.concatenate(valid_title_indices.values)
        df_no_titles = df.drop(index=title_indices_flat)
    else:
        df_no_titles = df

    target_col = 'heading_level' if 'heading_level' in df_no_titles.columns and df_no_titles['heading_level'].notna().any() else 'label'

    final_df = df_no_titles[df_no_titles[target_col].isin(TARGET_LABELS)].copy()
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(output_path, index=False)
    logger(f"✅ Cleaned data with {len(final_df)} heading samples saved to '{output_path}'")
    return final_df, target_col


def train_final_model(data_path: Path, target_col: str):
    logger("\n--- Starting Model Training ---")

    logger("   Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
    text_iterator = pd.read_csv(data_path, chunksize=CHUNK_SIZE, usecols=['text'], iterator=True)
    vectorizer.fit(pd.concat(text_iterator)['text'].astype(str))

    numerical_features = [
        'font_size', 'font_size_rank', 'relative_font_size', 'bold', 'italic', 'underline',
        'caps_ratio', 'word_count', 'title_case_ratio', 'has_numbering', 'punctuation',
        'whitespace_above', 'whitespace_below', 'bbox_width', 'bbox_height', 'relative_position'
    ]
    categorical_features = ['alignment']
    label_map = {label: i for i, label in enumerate(TARGET_LABELS)}

    logger("   Training LightGBM model on data chunks...")
    model = None
    chunk_iterator = pd.read_csv(data_path, chunksize=CHUNK_SIZE, iterator=True)
    
    # --- SAVE DUMMY COLUMNS (THE FIX) ---
    # Get dummy columns from the first chunk to establish the structure
    first_chunk = next(chunk_iterator)
    dummy_cols_df = pd.get_dummies(first_chunk[categorical_features], sparse=True, dtype=float)
    DUMMY_COLUMNS = dummy_cols_df.columns.tolist()
    joblib.dump(DUMMY_COLUMNS, MODEL_DIR / 'dummy_columns.joblib')
    logger(f"   Saved dummy column structure with {len(DUMMY_COLUMNS)} columns.")
    
    # Process the first chunk
    y_train = first_chunk[target_col].map(label_map)
    num_feat = csr_matrix(first_chunk[numerical_features].values)
    cat_feat = dummy_cols_df.sparse.to_coo()
    text_feat = vectorizer.transform(first_chunk['text'].astype(str))
    X_train = hstack([num_feat, cat_feat, text_feat], format='csr')
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    model = lgb.train({'objective': 'multiclass', 'num_class': 3, 'verbose': -1},
                      lgb_train, num_boost_round=15, keep_training_booster=True)
    logger("   Trained on chunk 1...")
    
    validation_chunk = first_chunk
    i = 1
    # Process remaining chunks
    for chunk in chunk_iterator:
        i += 1
        y_train = chunk[target_col].map(label_map)
        num_feat = csr_matrix(chunk[numerical_features].values)
        cat_feat_df = pd.get_dummies(chunk[categorical_features], sparse=True, dtype=float)
        cat_feat = cat_feat_df.reindex(columns=DUMMY_COLUMNS, fill_value=0).sparse.to_coo()
        text_feat = vectorizer.transform(chunk['text'].astype(str))
        X_train = hstack([num_feat, cat_feat, text_feat], format='csr')
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        model = lgb.train({'objective': 'multiclass', 'num_class': 3, 'verbose': -1},
                          lgb_train, num_boost_round=15, init_model=model, keep_training_booster=True)
        logger(f"   Trained on chunk {i}...")
        validation_chunk = chunk
    
    logger("\n📊 Evaluating model on final chunk...")
    y_val = validation_chunk[target_col].map(label_map).dropna()
    valid_indices = y_val.index
    
    num_feat_val = csr_matrix(validation_chunk.loc[valid_indices, numerical_features].values)
    cat_feat_val_df = pd.get_dummies(validation_chunk.loc[valid_indices, categorical_features], sparse=True, dtype=float)
    cat_feat_val = cat_feat_val_df.reindex(columns=DUMMY_COLUMNS, fill_value=0).sparse.to_coo()
    text_feat_val = vectorizer.transform(validation_chunk.loc[valid_indices, 'text'].astype(str))
    X_val = hstack([num_feat_val, cat_feat_val, text_feat_val], format='csr')

    y_pred = np.argmax(model.predict(X_val), axis=1)
    print(classification_report(y_val, y_pred, target_names=TARGET_LABELS))

    logger(f"💾 Saving artifacts to '{MODEL_DIR}/'...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(str(MODEL_DIR / 'lgbm_model.txt'))
    joblib.dump(vectorizer, MODEL_DIR / 'tfidf_vectorizer.joblib')
    joblib.dump(numerical_features, MODEL_DIR / 'numerical_features.joblib')
    joblib.dump(categorical_features, MODEL_DIR / 'categorical_features.joblib')
    joblib.dump(label_map, MODEL_DIR / 'label_map.joblib')
    logger("✅ Training complete and all artifacts saved.")

if __name__ == '__main__':
    _, target_column = clean_data_for_training(SOURCE_CSV_FILE, CLEANED_FOR_TRAINING_CSV)
    train_final_model(CLEANED_FOR_TRAINING_CSV, target_column)

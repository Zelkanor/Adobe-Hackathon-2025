import pandas as pd
import os

def append_docbank_to_doclay(docbank_csv, doclay_csv, output_csv):
    print("Loading datasets...")
    doclay_df = pd.read_csv(doclay_csv)
    docbank_df = pd.read_csv(docbank_csv)

    if os.path.exists(output_csv):
        print("Appending to existing merged file...")
        existing = pd.read_csv(output_csv)
        merged_df = pd.concat([existing, docbank_df], ignore_index=True)
    else:
        print("Merging DocLayNet and DocBank...")
        merged_df = pd.concat([doclay_df, docbank_df], ignore_index=True)

    merged_df.to_csv(output_csv, index=False)
    print(f"Merged file saved: {output_csv} (Total rows: {len(merged_df)})")

if __name__ == '__main__':
    docbank_csv = r"C:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Round-1A\dataset_prep\docbank_group.csv"
    doclay_csv = r"C:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Round-1A\dataset_prep\doclaynet_group.csv"
    output_csv = r"C:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Round-1A\dataset_prep\merged_features.csv"

    append_docbank_to_doclay(docbank_csv, doclay_csv, output_csv)


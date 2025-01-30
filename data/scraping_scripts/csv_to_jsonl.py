import json
import uuid

import pandas as pd
import tiktoken


# Function to count tokens using tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(
        encoding.encode(
            string, disallowed_special=(encoding.special_tokens_set - {"<|endoftext|>"})
        )
    )
    return num_tokens


# Function to clean or remove specific content, e.g., copyright headers
def remove_copyright_header(content: str) -> str:
    # Implement any cleaning logic you need here
    return content


# Function to convert DataFrame to JSONL format with token counting
def convert_to_jsonl_with_conditions(df, encoding_name="cl100k_base"):
    jsonl_data = []
    for _, row in df.iterrows():
        token_count = num_tokens_from_string(row["text"], encoding_name)

        # Skip entries based on token count conditions
        if token_count < 100 or token_count > 200_000:
            print(f"Skipping {row['title']} due to token count {token_count}")
            continue

        cleaned_content = remove_copyright_header(row["text"])

        entry = {
            "tokens": token_count,  # Token count using tiktoken
            "doc_id": str(uuid.uuid4()),  # Generate a unique UUID
            "name": row["title"],
            "url": row["tai_url"],
            "retrieve_doc": (token_count <= 8000),  # retrieve_doc condition
            "source": "tai_blog",
            "content": cleaned_content,
        }
        jsonl_data.append(entry)
    return jsonl_data


# Load the CSV file
data = pd.read_csv("data/tai.csv")

# Convert the dataframe to JSONL format with token counting and conditions
jsonl_data_with_conditions = convert_to_jsonl_with_conditions(data)

# Save the output to a new JSONL file using json.dumps to ensure proper escaping
output_path = "data/tai_blog_data_conditions.jsonl"
with open(output_path, "w") as f:
    for entry in jsonl_data_with_conditions:
        f.write(json.dumps(entry) + "\n")

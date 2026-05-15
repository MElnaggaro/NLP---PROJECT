import os
from convokit import download

# Subreddits used for the classification dataset
subreddits = [
    "subreddit-Republican",
    "subreddit-democrats"
]

# Directory where the datasets will be downloaded
output_dir = os.path.join(os.getcwd(), "reddit_project_data")
os.makedirs(output_dir, exist_ok=True)

print("Starting dataset download using ConvoKit...\n" + "-" * 50)

for subreddit in subreddits:
    try:
        print(f"Downloading: {subreddit} ... (this may take some time)")
        
        downloaded_path = download(
            subreddit,
            data_dir=output_dir
        )
        
        print(f"Successfully downloaded and extracted to: {downloaded_path}\n")
    
    except Exception as e:
        print(f"Error while downloading {subreddit}: {e}\n")

print("-" * 50)
print("Download completed successfully. Dataset is ready.")
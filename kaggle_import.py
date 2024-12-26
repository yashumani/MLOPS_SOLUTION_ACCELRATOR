from kaggle.api.kaggle_api_extended import KaggleApi
import datasets  # Hugging Face Datasets

def fetch_kaggle_dataset(dataset_name, save_path="datasets/"):
    """Fetch datasets from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=save_path, unzip=True)
    print(f"Dataset {dataset_name} downloaded to {save_path}.")

def fetch_huggingface_dataset(dataset_name):
    """Fetch datasets from Hugging Face."""
    dataset = datasets.load_dataset(dataset_name)
    print(f"Dataset {dataset_name} loaded.")
    return dataset

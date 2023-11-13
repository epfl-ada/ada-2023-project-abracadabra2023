import os
import requests
import tarfile


def download_and_extract(url, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get the file name from the URL
    file_name = os.path.join(destination_folder, url.split("/")[-1])

    # Download the file
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    print("Download complete")
    print("copying to file")
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)
    print("File copied")

    # Extract the contents of the tar.gz file*
    print("Extracting the archive")
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(destination_folder)

    # Go through the data folder and copy the files from subfolders to the data folder
    for root, dirs, files in os.walk(destination_folder):
        for name in files:
            if root != destination_folder:
                # if the file already exists in the data folder, overwrite it
                if os.path.exists(os.path.join(destination_folder, name)):
                    os.remove(os.path.join(destination_folder, name))
                os.rename(
                    os.path.join(root, name), os.path.join(destination_folder, name)
                )
            # remove the empty subfolders
            if not os.listdir(root):
                os.rmdir(root)

    print(f"Archive downloaded and extracted to {destination_folder}")


if __name__ == "__main__":
    # URL of the archive
    archive_url = (
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz"
    )

    # Destination folder for extraction
    destination_folder = "data"

    # Download and extract the archive
    download_and_extract(archive_url, destination_folder + "/graph")
    # download_and_extract("https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz", destination_folder)
    # download_and_extract("https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_html.tar.gz", destination_folder)

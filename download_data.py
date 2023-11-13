import os
import requests
import tarfile


def download_and_extract(url, destination_folder, subfolder=None):
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
    eextracted_folder_name = None
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(destination_folder)
        # get the name of the extracted folder
        extracted_folder_name = os.path.commonprefix(tar.getnames())

    # rename the folder to subfolder
    if subfolder is not None and extracted_folder_name is not None:
        if os.path.exists(os.path.join(destination_folder, subfolder)):
            for file in os.listdir(os.path.join(destination_folder, subfolder)):
                os.remove(os.path.join(destination_folder, subfolder, file))
            os.rmdir(os.path.join(destination_folder, subfolder))
        os.rename(
            os.path.join(destination_folder, extracted_folder_name),
            os.path.join(destination_folder, subfolder),
        )

    print(f"Archive downloaded and extracted to {destination_folder}")


if __name__ == "__main__":
    # Destination folder for extraction
    destination_folder = "data"
    # Download and extract the archive
    download_and_extract(
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz",
        destination_folder,
        subfolder="graph",
    )
    download_and_extract(
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz",
        destination_folder,
        subfolder="articles_plain_text",
    )
    # download_and_extract(
    #     "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_html.tar.gz",
    #     destination_folder,
    #     subfolder="articles_html",
    # )

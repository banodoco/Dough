import os
import random
import string
import tarfile
import time
import zipfile
import requests
import streamlit as st
from utils.common_decorators import with_refresh_lock
from utils.state_refresh import refresh_app


@with_refresh_lock
def download_file_widget(url, filename, dest):
    save_directory = dest
    zip_filename = filename
    filepath = os.path.join(save_directory, zip_filename)

    # ------- deleting partial downloads
    if st.session_state.get("delete_partial_download", None):
        fp = st.session_state["delete_partial_download"]
        st.session_state["delete_partial_download"] = None
        if os.path.exists(fp):
            os.remove(fp)
            st.info("Partial downloads deleted")
            time.sleep(0.3)
            refresh_app()

    # checking if the file already exists
    if os.path.exists(os.path.join(dest, filename)):
        st.warning("File already present")
        time.sleep(1)
        refresh_app()

    # setting this file for deletion, incase it's not downloaded properly
    # if it is downloaded properly then it will be removed from here (all these steps because of streamlit!)
    st.session_state["delete_partial_download"] = filepath

    with st.spinner("Downloading model..."):
        download_bar = st.progress(0, text="")
        os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

        # Download the model and save it to the directory
        response = requests.get(url, stream=True)
        cancel_download = False

        if st.button("Cancel"):
            st.session_state["delete_partial_download"] = filepath

        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            total_size_mb = total_size / (1024 * 1024)

            start_time = time.time()

            with open(filepath, "wb") as f:
                received_bytes = 0
                for data in response.iter_content(chunk_size=1048576):
                    if cancel_download:
                        raise Exception("download cancelled")

                    f.write(data)
                    received_bytes += len(data)
                    progress = received_bytes / total_size
                    received_mb = received_bytes / (1024 * 1024)

                    elapsed_time = time.time() - start_time
                    download_speed = received_bytes / elapsed_time / (1024 * 1024)

                    download_bar.progress(
                        progress,
                        text=f"Downloaded: {received_mb:.2f} MB / {total_size_mb:.2f} MB | Speed: {download_speed:.2f} MB/sec",
                    )

            st.success(f"Downloaded {filename} and saved to {save_directory}")
            time.sleep(1)
            download_bar.empty()

            if url.endswith(".zip") or url.endswith(".tar"):
                st.success("Extracting the zip file. Please wait...")
                new_filepath = filepath.replace(zip_filename, "")
                if url.endswith(".zip"):
                    with zipfile.ZipFile(f"{filepath}", "r") as zip_ref:
                        zip_ref.extractall(new_filepath)
                else:
                    with tarfile.open(f"{filepath}", "r") as tar_ref:
                        tar_ref.extractall(new_filepath)

                os.remove(filepath)
            else:
                os.rename(filepath, filepath.replace(zip_filename, filename))
                print("removing ---------")

            st.session_state["delete_partial_download"] = None
        else:
            st.error("Unable to access model url")
            time.sleep(1)

        refresh_app()

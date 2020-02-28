from typing import Union, Iterator, Callable, Any, Optional, Dict
import os, sys, re
import gzip


"""helper functions to abstract access to Azure blobs"""


def _try_parse_azure_blob_uri(path: str):
    try:
        m = re.compile("https://([a-z0-9]*).blob.core.windows.net/([^/]*)/(.*)").match(path)
        #print (m.group(1))
        #print (m.group(2))
        #print (m.group(3))
        return (m.group(1), m.group(2), m.group(3))
    except:
        return None


def _get_azure_key(storage_account: str, credentials: Optional[Union[str,Dict[str,str]]]):
    if not credentials:
        return None
    elif isinstance(credentials, str):
        return credentials
    else:
        return credentials[storage_account]


def read_utf8_file(path: str, credentials: Optional[Union[str,Dict[str,str]]]) -> Iterator[str]:
    blob_data = _try_parse_azure_blob_uri(path)
    if blob_data is None:
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return f.read()
        else:
            raise NotImplementedError  # @TODO
    else:
        try:
            # pip install azure-storage-blob
            from azure.storage.blob import BlobClient
        except:
            print("Failed to import azure.storage.blob. Please pip install azure-storage-blob", file=sys.stderr)
            raise
        data = BlobClient.from_blob_url(path, credential=_get_azure_key(storage_account=blob_data[0], credentials=credentials)).download_blob().readall()
        if path.endswith('.gz'):
            data = gzip.decompress(data)
        return data.decode(encoding='utf-8')


def find_files(dir: str, ext: str, credentials: Optional[Union[str,Dict[str,str]]]):
    blob_data = _try_parse_azure_blob_uri(dir)
    if blob_data is None:
        return [os.path.join(dir, path.name)
                for path in os.scandir(dir)
                if path.is_file() and (ext is None or path.name.endswith(ext))]
    else:
        try:
            # pip install azure-storage-blob
            from azure.storage.blob import ContainerClient
        except:
            print("Failed to import azure.storage.blob. Please pip install azure-storage-blob", file=sys.stderr)
            raise
        account, container, blob_path = blob_data

        print("find_files: enumerating blobs in", dir, file=sys.stderr, flush=True)
        # @BUGBUG: The prefix does not seem to have to start; seems it can also be a substring
        container_uri = "https://" + account + ".blob.core.windows.net/" + container
        container_client = ContainerClient.from_container_url(container_uri, credential=_get_azure_key(account, credentials))
        if not blob_path.endswith("/"):
            blob_path += "/"
        blob_uris = [container_uri + "/" + blob["name"] for blob in container_client.walk_blobs(blob_path, delimiter="") if (ext is None or blob["name"].endswith(ext))]
        print("find_files:", len(blob_uris), "blobs found", file=sys.stderr, flush=True)
        for blob_name in blob_uris[:10]:
            print(blob_name, file=sys.stderr, flush=True)
        return blob_uris

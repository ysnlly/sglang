"""Download the sixteen public source images and verify the measured byte hashes.

Sources: scikit-image v0.25.2, commit e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae.
See its skimage/data/_registry.py for checksums and _fetchers.py for attribution.
This package distributes a downloader, not the images themselves.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import requests

REVISION = 'e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae'
BASE_URL = f'https://raw.githubusercontent.com/scikit-image/scikit-image/{REVISION}/skimage/data/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name('input_images.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []
    with requests.Session() as session:
        for item in contract:
            name = item['name']
            if Path(name).name != name:
                raise ValueError('Invalid image filename')
            url = BASE_URL + name
            response = session.get(url, timeout=(10, 90))
            response.raise_for_status()
            blob = response.content
            digest = hashlib.sha256(blob).hexdigest()
            if digest != item['original_sha256']:
                raise ValueError(f'Source byte mismatch: {name}')
            with Image.open(io.BytesIO(blob)) as image:
                if list(image.size) != item['original_size']:
                    raise ValueError(f'Source dimension mismatch: {name}')
            (args.output / name).write_bytes(blob)
            receipts.append(dict(name=name, url=url, sha256=digest, bytes=len(blob)))
            print(json.dumps(receipts[-1]), flush=True)
    receipt = dict(revision=REVISION, images=receipts, image_count=len(receipts),
                   all_source_hashes_passed=True, model_requests_sent=0)
    (args.output / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()

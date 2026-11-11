"""Export the pinned complete MMMU corpus with portable paths and exact hashes.

Uses the measured export's RGB/PNG conversion, subject order and question fields.
Only image paths are made relative. A semantic hash binds all 900 full records.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq


def semantic_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, help='Reuse previously verified/downloaded parquet cache')
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name('mmmu_contract.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / 'images'
    images.mkdir()
    files = sorted(contract['parquet_sha256'])
    assert len(files) == 30

    def download(name):
        path = hf_hub_download(contract['dataset'], name, repo_type='dataset',
                               revision=contract['revision'], cache_dir=str(args.cache_dir or args.output / 'hf-cache'))
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == contract['parquet_sha256'][name], name
        return name, path

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, path in pool.map(download, files):
            for row in pq.read_table(path).to_pylist():
                converted = []
                for index in range(1, 8):
                    source = row.get(f'image_{index}')
                    if source is None:
                        continue
                    assert isinstance(source, dict) and source.get('bytes')
                    filename = f"{row['id']}_{index}.png"
                    assert Path(filename).name == filename
                    destination = images / filename
                    assert not destination.exists()
                    with Image.open(io.BytesIO(source['bytes'])) as image:
                        image.convert('RGB').save(destination)
                    converted.append(dict(index=index, path='images/' + filename,
                                          sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
                assert converted
                options = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
                assert isinstance(options, list)
                rows.append(dict(id=row['id'], subject=name.split('/')[0], question=row['question'],
                                 options=options, answer=row['answer'], question_type=row['question_type'],
                                 images=converted))
            print(json.dumps(dict(subject=name.split('/')[0], samples_so_far=len(rows))), flush=True)
    rows.sort(key=lambda row: row['id'])
    assert len(rows) == len({row['id'] for row in rows}) == 900
    assert len({row['subject'] for row in rows}) == 30
    assert sum(len(row['images']) > 1 for row in rows) == 43
    (args.output / 'samples.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    observed = semantic_digest(rows)
    assert observed == contract['portable_semantic_sha256'], ('Export differs from measured corpus', observed)
    manifest = dict(contract, samples_sha256=hashlib.sha256((args.output / 'samples.json').read_bytes()).hexdigest(),
                    path_semantics='Image paths are relative to this export root.', image_files=sum(len(row['images']) for row in rows))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='passed', questions=900, images=manifest['image_files'],
                          portable_semantic_sha256=observed, requests_sent=0)), flush=True)


if __name__ == '__main__':
    main()

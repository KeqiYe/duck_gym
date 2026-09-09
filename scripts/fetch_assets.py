"""Fetch hash-pinned upstream files independently, so interrupted downloads can resume."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import base64
from http.client import IncompleteRead
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'assets/microduck'


def fetch():
    meta = json.loads((ROOT / 'docs/microduck_asset_manifest.json').read_text())
    prefix = 'src/mjlab_microduck/robot/microduck/'
    files = [(m['path'], m['path'][len(prefix):], m['git_blob_sha']) for m in meta['meshes']]
    files += [(meta['model']['path'], 'robot_allcollisions.xml', meta['model']['git_blob_sha']),
              ('README.md', 'UPSTREAM_README.md', '3f2852e79b2d06443fe2c8920d9f8bfccdfbb566'),
              ('LICENSE', 'UPSTREAM_LICENSE', '5babf660c98eac4418422e1a62a5d9fc51a4d63e')]
    def digest(data):
        return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    def get(item):
        source, rel, expected = item
        target = DEST / rel
        if target.exists() and digest(target.read_bytes()) == expected:
            print(f'Verified {rel}', flush=True)
            return
        url = f"https://raw.githubusercontent.com/pollen-robotics/microduck_rl/{meta['commit']}/{source}"
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                request_url = url if attempt < 2 else f'https://api.github.com/repos/pollen-robotics/microduck_rl/git/blobs/{expected}'
                request = urllib.request.Request(request_url, headers={'User-Agent': 'duck-gym-asset-fetcher'})
                with urllib.request.urlopen(request, timeout=40) as response:
                    data = response.read()
                if attempt >= 2:
                    data = base64.b64decode(json.loads(data)['content'])
                if digest(data) != expected:
                    raise ValueError(f'Blob hash mismatch: {rel}')
                temporary = target.with_suffix(target.suffix+'.download')
                temporary.write_bytes(data)
                temporary.replace(target)
                print(f'Downloaded {rel}', flush=True)
                return
            except (OSError, ValueError, IncompleteRead) as error:
                if attempt == 3:
                    raise RuntimeError(f'Download failed: {rel}') from error
                time.sleep(1+attempt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get, item) for item in files]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('; '.join(failures))
    (DEST / 'PROVENANCE.json').write_text(json.dumps({
        'repository': meta['repository'], 'commit': meta['commit'],
        'verified_files': len(files), 'license': meta['upstream_license_summary'],
    }, indent=2)+'\n')
    print(f'All {len(files)} pinned files verified at {DEST}', flush=True)


if __name__ == '__main__':
    fetch()

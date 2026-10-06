#!/usr/bin/env python
"""NOT USED FOR THE FORMAL RUN (2026-10-04): this draft followed the upstream AerialDetection devkit (gap 200, PNG
tiles) and was stopped. The formal tiles were produced by the authors' own EAV-DETR/AerialDetection/DOTA_devkit/
prepare_dota1.py at commit fb99da7b (gap 512, 28 windows per 3840 x 2160 frame, JPEG tiles, 12-class filter); see HANDOFF.md.

Original draft docstring: Tile the CODrone frames for the EAV-DETR system-level comparison (2026-10-04).

Follows the EAV-DETR README route (AerialDetection DOTA_devkit, ImgSplit_multi_process at the cloned commit): every
official split (train, val, test) is cut into 1024 x 1024 windows with gap 200, rate 1, IOF threshold 0.7, best point
order and zero padding; tiles and their DOTA labels go to CODrone1024/{train1024,val1024,test1024}/{images,labelTxt}, the
layout the official configs read. Only one deviation from running the devkit unchanged: the devkit reads and writes with
the same extension, while the CODrone frames are JPEG; here frames are read as .jpg and tiles written as lossless .png,
so tile pixels equal the decoded frame pixels. Window geometry, label clipping, difficulty-2 marking and padding are the
devkit's own code. Afterwards every split is checked (one label per tile, 1024 x 1024 images, expected 15 windows per
3840 x 2160 frame) and a manifest with sizes and SHA-256 is written.

Usage: DOTA_DEVKIT=DIR split_codrone.py --devkit DIR --src DATA/CODrone --dst DATA/CODrone1024 --procs 96 --receipt FILE
"""
import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

DEVKIT = os.environ.get('DOTA_DEVKIT')
if DEVKIT:
    sys.path.insert(0, DEVKIT)
import ImgSplit_multi_process as ims  # noqa: E402  (the devkit's own splitter)


class SplitJpgToPng(ims.splitbase):
    """The devkit splitter with one change: frames are read as .jpg, tiles are written with self.ext (.png)."""

    def splitdata(self, rate):
        imagelist = ims.GetFileFromThisRootDir(self.imagepath)
        names = [ims.util.custombasename(x) for x in imagelist if ims.util.custombasename(x) != 'Thumbs']
        worker = partial(self.SplitSingle, rate=rate, extent='.jpg')
        self.pool.map(worker, names)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def windows(width, height, subsize=1024, gap=200):
    """The devkit's SplitSingle loop, reproduced only to predict the number of windows."""
    slide = subsize - gap
    out, left = [], 0
    while left < width:
        if left + subsize >= width:
            left = max(width - subsize, 0)
        up = 0
        while up < height:
            if up + subsize >= height:
                up = max(height - subsize, 0)
            out.append((left, up))
            if up + subsize >= height:
                break
            up += slide
        if left + subsize >= width:
            break
        left += slide
    return out


def file_record(path):
    return os.path.basename(path), os.path.getsize(path), sha256(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--devkit', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--procs', type=int, default=96)
    ap.add_argument('--receipt', required=True)
    a = ap.parse_args()
    if os.path.exists(a.receipt):
        raise SystemExit('receipt exists; refusing to overwrite')
    assert DEVKIT and os.path.abspath(DEVKIT) == os.path.abspath(a.devkit), 'set DOTA_DEVKIT to --devkit'
    os.makedirs(a.dst, exist_ok=True)
    t0 = time.time()
    receipt = dict(schema='rotcert.codrone-eav-tiles.v1', devkit=a.devkit, params=dict(gap=200, subsize=1024, rate=1,
                   thresh=0.7, choosebestpoint=True, padding=True, read_ext='.jpg', write_ext='.png'), splits={})
    for split in ('train', 'val', 'test'):
        src, dst = os.path.join(a.src, split), os.path.join(a.dst, split + '1024')
        if os.path.exists(dst):
            raise SystemExit(f'{dst} exists; refusing to overwrite')
        frames = sorted(os.listdir(os.path.join(src, 'images')))
        s = SplitJpgToPng(src, dst, gap=200, subsize=1024, thresh=0.7, choosebestpoint=True, ext='.png',
                          padding=True, num_process=a.procs)
        s.splitdata(1)
        s.pool.close()
        s.pool.join()
        tiles = sorted(os.listdir(os.path.join(dst, 'images')))
        labels = sorted(os.listdir(os.path.join(dst, 'labelTxt')))
        stems_t = {os.path.splitext(t)[0] for t in tiles}
        stems_l = {os.path.splitext(t)[0] for t in labels}
        # expected windows from each frame's own size
        expected = 0
        with Pool(a.procs) as pool:
            dims = pool.map(_dims, [os.path.join(src, 'images', f) for f in frames])
        for w, h in dims:
            expected += len(windows(w, h))
        with Pool(a.procs) as pool:
            recs = pool.map(file_record, [os.path.join(dst, 'images', t) for t in tiles], chunksize=64)
        with Pool(a.procs) as pool:
            lrecs = pool.map(file_record, [os.path.join(dst, 'labelTxt', t) for t in labels], chunksize=256)
        with Pool(a.procs) as pool:
            tdims = pool.map(_dims, [os.path.join(dst, 'images', t) for t in tiles[::97]])
        man = os.path.join(a.dst, f'{split}1024-MANIFEST.jsonl')
        with open(man, 'w') as f:
            for name, size, digest in recs:
                f.write(json.dumps(dict(kind='image', name=name, bytes=size, sha256=digest)) + '\n')
            for name, size, digest in lrecs:
                f.write(json.dumps(dict(kind='label', name=name, bytes=size, sha256=digest)) + '\n')
        receipt['splits'][split] = dict(frames=len(frames), tiles=len(tiles), label_files=len(labels),
                                        expected_windows=expected, every_tile_has_label=stems_t == stems_l,
                                        sampled_tile_dims=sorted(set(map(tuple, tdims))),
                                        tile_bytes=sum(r[1] for r in recs), manifest=man, manifest_sha256=sha256(man))
        print(split, json.dumps(receipt['splits'][split]), flush=True)
        assert stems_t == stems_l and len(tiles) == expected, split
    receipt['seconds'] = time.time() - t0
    with open(a.receipt, 'w') as f:
        json.dump(receipt, f, indent=1)
    print('receipt', a.receipt, sha256(a.receipt))


def _dims(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (img.shape[1], img.shape[0])


if __name__ == '__main__':
    main()

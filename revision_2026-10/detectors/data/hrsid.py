#!/usr/bin/env python3
"""HRSID preparation with explicit annotation version and panorama-disjoint splits."""
import argparse
from collections import Counter
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'export'))
from common import atomic_json, write_jsonl, sha256, utc
from canonical import polygon_obb
from stage import (no_overlap, images, snapshot, digest, file_record, decode_image,
                   dota_objects, verify_staged_data, INVENTORY_VERSION)
RAR_SHA = '743be02ed6df0a303fecd15db237ad84ba6fbf1b09b5111f171196e92ec2444d'
ZIP_SHA = 'f42117babc7be3699e65848368a53dfbb9fd0d64dd5353f22f68c530d3713071'
ANNOTATION_HASHES = {
    'train2017.json': 'b475bd3842b1a26f8f41a90d54406b633a0a8dce0afb2feb5b5a7ea3d77030b0',
    'test2017.json': 'ca2ec30f30e377aa1667892d0fd69ccb912c09446484478d938525393ea1de7e',
    'train_test2017.json': 'b4805aa29c3a54e580df67687106ccb0e3a181b750fc9fdc830a656037803645',
    'inshore.json': '4deff21b5c2785e10cb84f1d73dfc5e287982c816f5d0b406b185f3f7bd719cb',
    'offshore.json': '0990177d9ff9ded17451604a0592e500e93432670b859c3572272290fbc9a831'}

def verified_json(path, name):
    content = snapshot(path)
    if digest(content) != ANNOTATION_HASHES[name]:
        raise ValueError('Unverified annotation bytes: ' + name)
    return json.loads(content)

def annotation_multiset(doc):
    ids = {i['id']: Path(i['file_name']).stem for i in doc['images']}
    categories = {c['id']: c['name'].lower() for c in doc['categories']}
    rows = []
    for ann in doc['annotations']:
        rec = {k: v for k, v in ann.items() if k not in ('id', 'image_id', 'category_id')}
        rec.update(image=ids[ann['image_id']], category=categories[ann['category_id']])
        rows.append(json.dumps(rec, sort_keys=True, allow_nan=False))
    return Counter(rows)

def shore_partition(inshore, offshore, combined, expected_ids):
    groups = {}
    for name, doc in [('inshore', inshore), ('offshore', offshore)]:
        ids = [Path(i['file_name']).stem for i in doc['images']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate shore image')
        if set(ids) & set(groups):
            raise ValueError('Shore groups overlap')
        groups.update({i: name for i in ids})
    if set(groups) != set(expected_ids):
        raise ValueError('Shore groups must cover all selected images')
    if annotation_multiset(inshore) + annotation_multiset(offshore) != annotation_multiset(combined):
        raise ValueError('Shore annotations differ from combined annotation version')
    return groups

VERSION_COUNTS = {
    'combined-train-test': (5604, 16951, 12),
    'train-plus-test': (5604, 16969, 13),
}

GEOMETRY_POLICY = {
    'id': 'hrsid-mask-obb-finite-zero-area-v1',
    'coordinates': 'continuous source mask vertices; convex-hull OBB as before',
    'exclusion': 'all mask vertices exactly collinear or coincident (zero convex-hull area)',
    'arithmetic': 'exact rational test of source JSON numbers, no epsilon or rounding threshold',
    'reason': 'finite_zero_area_polygon',
    'bbox_fallback': False,
    'pixel_expansion': False,
    'positive_but_unrepresentable': 'reject; never reclassify as zero area',
    'malformed_or_nonfinite': 'reject',
    'images_and_panorama_splits': 'retain all source images and the unchanged split',
}

def mask_obb(seg):
    # Positive pixel area/HBB does not give collinear continuous vertices a
    # positive OBB. Pixel expansion would change the existing protocol.
    if (not isinstance(seg, list) or not seg or
            any(not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2
                for poly in seg)):
        raise ValueError('HRSID requires actual polygon masks, no bbox fallback')
    coords = [v for poly in seg for v in poly]
    try:
        finite = all(type(v) in (int, float) and math.isfinite(v) for v in coords)
    except (OverflowError, TypeError):
        finite = False
    if not finite:
        raise ValueError('Invalid polygon: HRSID mask coordinates must be finite numbers')
    points = list(zip(coords[::2], coords[1::2]))
    origin = points[0]
    other = next((point for point in points if point != origin), None)
    if other is None:
        return None
    x, y = map(Fraction, origin)
    dx, dy = Fraction(other[0]) - x, Fraction(other[1]) - y
    if all(dx * (Fraction(py) - y) == dy * (Fraction(px) - x) for px, py in points):
        return None
    # A positive polygon that collapses in float32 must still fail closed.
    return polygon_obb(coords)

def geometry_audit(ims, anns, parts, shore=None):
    """Conserve source counts and record each excluded source annotation."""
    shore = shore or {}
    split_of = {image: split for split, ids in parts.items() for image in ids}
    if set(split_of) != set(ims) or sum(map(len, parts.values())) != len(ims):
        raise ValueError('Geometry audit split must preserve all source images exactly once')
    if shore and (set(shore) != set(ims) or set(shore.values()) - {'inshore', 'offshore'}):
        raise ValueError('Geometry audit requires complete verified shore membership')
    def counts(images):
        return {'images': images, 'source_annotations': 0, 'valid_annotations': 0, 'excluded_annotations': 0}
    by_split = {split: counts(len(ids)) for split, ids in parts.items()}
    by_shore = {group: counts(sum(v == group for v in shore.values()))
                for group in ('inshore', 'offshore')} if shore else {}
    by_document = {}
    exclusions = []
    for ann in anns:
        image = ann['image_id']
        if image not in split_of:
            raise ValueError('Geometry audit annotation image is not in split')
        document = ann['source_document']
        group = shore.get(image)
        buckets = [by_split[split_of[image]], by_document.setdefault(document,
            {'source_annotations': 0, 'valid_annotations': 0, 'excluded_annotations': 0})]
        if shore:
            buckets.append(by_shore[group])
        excluded = ann['obb'] is None
        for bucket in buckets:
            bucket['source_annotations'] += 1
            bucket['excluded_annotations' if excluded else 'valid_annotations'] += 1
        if excluded:
            record = {key: ann[key] for key in ('image_id', 'source_image_id',
                'source_annotation_id', 'source_document', 'class', 'source_segmentation')}
            record.update(split=split_of[image], shore_group=group, reason=GEOMETRY_POLICY['reason'])
            exclusions.append(record)
    for bucket in list(by_split.values()) + list(by_shore.values()) + list(by_document.values()):
        if bucket['source_annotations'] != bucket['valid_annotations'] + bucket['excluded_annotations']:
            raise ValueError('HRSID geometry count conservation failed')
    return {'policy': dict(GEOMETRY_POLICY), 'source_images': len(ims),
            'source_annotations': len(anns), 'valid_annotations': len(anns) - len(exclusions),
            'excluded_annotations': len(exclusions), 'counts_by_split': by_split,
            'shore_status': 'VERIFIED_IMAGE_MEMBERSHIP' if shore else 'NOT_SUPPLIED',
            'counts_by_shore': by_shore, 'counts_by_source_document': by_document,
            'exclusions': exclusions}

def combine_coco(documents, source_names=None):
    if source_names is None:
        source_names = ['document-' + str(i) for i in range(len(documents))]
    if len(source_names) != len(documents) or len(set(source_names)) != len(source_names):
        raise ValueError('Annotation source names must identify each document once')
    images_by_name, anns = {}, []
    for source_name, doc in zip(source_names, documents):
        idmap = {}
        for image in doc['images']:
            name = Path(image['file_name']).stem
            if image['id'] in idmap or name in images_by_name:
                raise ValueError('Duplicate COCO image or filename')
            idmap[image['id']] = name
            images_by_name[name] = image
        seen = set()
        categories = {c['id']: c['name'].lower() for c in doc['categories']}
        for ann in doc['annotations']:
            if ann['id'] in seen:
                raise ValueError('Duplicate COCO annotation id')
            seen.add(ann['id'])
            if ann['image_id'] not in idmap:
                raise ValueError('Unknown annotation image')
            if categories.get(ann['category_id']) != 'ship':
                raise ValueError('Unexpected HRSID category')
            seg = ann.get('segmentation')
            box = mask_obb(seg)
            record = {'image_id': idmap[ann['image_id']], 'obb': box, 'class': 'ship',
                      'source_annotation_id': ann['id'], 'source_image_id': ann['image_id'],
                      'source_document': source_name}
            if box is None:
                record['source_segmentation'] = seg
            anns.append(record)
    return images_by_name, anns

def split_panoramas(image_ids, mapping, seed=0, train_fraction=.65):
    if set(image_ids) != set(mapping):
        raise ValueError('Scene map must cover exactly the selected images')
    if any(not isinstance(v, str) or not v for v in mapping.values()):
        raise ValueError('Missing panorama identity')
    if not 0 < train_fraction < 1:
        raise ValueError('Invalid train fraction')
    scenes = sorted(set(mapping.values()))
    if len(scenes) < 2:
        raise ValueError('Need at least two verified panoramas')
    random.Random(seed).shuffle(scenes)
    n = max(1, min(len(scenes) - 1, int(len(scenes) * train_fraction)))
    train = set(scenes[:n])
    parts = {'train': sorted(i for i in image_ids if mapping[i] in train), 'eval': sorted(i for i in image_ids if mapping[i] not in train)}
    no_overlap(parts, mapping)
    return parts

def index_prepared_outputs(output, parts, splits):
    """Index and decode actual prepared files; never synthesize image hashes.

    Main has already enforced official source/version counts. This helper also
    verifies annotation/GT object counts before binding the prepared inventory.
    """
    output=Path(output).resolve(); directories=[]
    if set(parts)!={'train','eval'} or set(splits)!=set(parts):
        raise ValueError('HRSID prepared split mismatch')
    if set(parts['train']) & set(parts['eval']):
        raise ValueError('HRSID prepared image overlap')
    for split,ids in parts.items():
        if not ids or len(ids)!=len(set(ids)) or splits[split]['images']!=len(ids):
            raise ValueError('HRSID prepared image count mismatch')
        image_dir=output/split/'images'; ann_dir=output/split/'annfiles'
        if set(images(image_dir))!=set(ids) or any(p.suffix!='.png' for p in images(image_dir).values()):
            raise ValueError('HRSID prepared image inventory mismatch')
        if {p.name for p in ann_dir.glob('*.txt')}!={i+'.txt' for i in ids}:
            raise ValueError('HRSID prepared annotation inventory mismatch')
        image_records=[]; annotation_records=[]; actual_counts=Counter()
        for image in ids:
            path=image_dir/(image+'.png'); content=snapshot(path)
            width,height=decode_image(content,path)
            image_records.append(dict(file_record(path,output,content),id=image,width=width,height=height))
            path=ann_dir/(image+'.txt'); content=snapshot(path)
            objects=dota_objects(path,'hrsid',content)
            actual_counts[image]=len(objects)
            annotation_records.append(dict(file_record(path,output,content),id=image))
        gt_path=output/('gt_'+split+'.jsonl'); gt_content=snapshot(gt_path)
        if digest(gt_content)!=splits[split]['gt_sha256']:
            raise ValueError('HRSID prepared ground truth changed')
        rows=[json.loads(line) for line in gt_content.decode('utf-8').splitlines() if line.strip()]
        gt_counts=Counter(row['image_id'] for row in rows)
        if not set(gt_counts)<=set(ids) or any(gt_counts[i]!=actual_counts[i] for i in ids) or len(rows)!=splits[split]['objects']:
            raise ValueError('HRSID prepared annotation/GT count mismatch')
        inv_path=output/('inventory_'+split+'.json')
        atomic_json(inv_path,{'version':INVENTORY_VERSION,'images':image_records,'annotations':annotation_records})
        splits[split].update(inventory_sha256=sha256(inv_path),image_bytes=sum(row['bytes'] for row in image_records))
        for kind,directory,suffix in [('images',image_dir,'.png'),('annotations',ann_dir,'.txt')]:
            directories.append({'directory':str(directory.relative_to(output)),'kind':kind,'suffix':suffix,'names':sorted(i+suffix for i in ids)})
    controls=[]
    for name in ['scene_map.json','split.json','shore_labels.json','geometry_audit.json']:
        path=output/name
        if name in ('shore_labels.json','geometry_audit.json') and not path.exists(): continue
        controls.append(file_record(path,output,snapshot(path)))
    return {'inventory_version':INVENTORY_VERSION,'root':str(output),'directories':directories,'controls':controls}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--scene-map', type=Path, required=True)
    p.add_argument('--annotation-choice', choices=['train-plus-test', 'combined-train-test'], required=True)
    p.add_argument('--annotations', nargs='+', type=Path, required=True, help='For train-plus-test: TRAIN then TEST; otherwise combined')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--shore-zip', type=Path)
    p.add_argument('--inshore', type=Path)
    p.add_argument('--offshore', type=Path)
    p.add_argument('--combined-for-shore', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--state-dir', type=Path, required=True)
    a = p.parse_args()
    marker = a.state_dir / 'data_markers/DATA_OK_hrsid'
    marker.unlink(missing_ok=True)
    archive_hash = sha256(a.archive)
    if (a.archive.stat().st_size, archive_hash) not in [(608910165, RAR_SHA), (614069765, ZIP_SHA)]:
        raise ValueError('HRSID archive differs from verified main record')
    names = ['train2017.json', 'test2017.json'] if a.annotation_choice == 'train-plus-test' else ['train_test2017.json']
    if len(a.annotations) != len(names):
        raise ValueError('Explicit annotation choice does not match files')
    docs = [verified_json(f, name) for f, name in zip(a.annotations, names)]
    ims, anns = combine_coco(docs, names)
    expected_images, expected, expected_excluded = VERSION_COUNTS[a.annotation_choice]
    if len(ims) != expected_images or len(anns) != expected:
        raise ValueError('HRSID selected version/count mismatch')
    shore_args = [a.shore_zip, a.inshore, a.offshore, a.combined_for_shore]
    if any(shore_args) and not all(shore_args):
        raise ValueError('Shore labels require ZIP, both groups and combined reference')
    shore = {}
    shore_provenance = {'status': 'NOT_SUPPLIED', 'note': 'RAR main COCO has no field; verified ZIP supplies separate subgroup JSONs'}
    if all(shore_args):
        if a.shore_zip.stat().st_size != 614069765 or sha256(a.shore_zip) != ZIP_SHA:
            raise ValueError('Shore labels require verified source ZIP')
        ins = verified_json(a.inshore, 'inshore.json')
        off = verified_json(a.offshore, 'offshore.json')
        combined = verified_json(a.combined_for_shore, 'train_test2017.json')
        if (len(ins['images']), len(ins['annotations']), len(off['images']), len(off['annotations'])) != (1031, 8206, 4573, 8745):
            raise ValueError('Shore count mismatch')
        shore = shore_partition(ins, off, combined, ims)
        shore_provenance = {'status': 'VERIFIED', 'source_archive_sha256': ZIP_SHA, 'json_sha256': {str(f): ANNOTATION_HASHES[name] for f,name in zip(shore_args[1:],['inshore.json','offshore.json','train_test2017.json'])}, 'reference_annotation_version': 'combined-train-test (16951)', 'usage': 'image membership only; training annotations retain explicit selected version'}
    mapping_content = snapshot(a.scene_map)
    mapping = json.loads(mapping_content)
    if len(set(mapping.values())) != 137:
        raise ValueError('Expected 137 verified source panoramas')
    parts = split_panoramas(ims, mapping)
    audit = geometry_audit(ims, anns, parts, shore)
    if (audit['source_annotations'] != expected or
            audit['excluded_annotations'] != expected_excluded or
            audit['valid_annotations'] != expected - expected_excluded):
        raise ValueError('HRSID selected geometry/version count mismatch')
    audit.update(annotation_choice=a.annotation_choice, archive_sha256=archive_hash,
                 annotation_hashes={name: ANNOTATION_HASHES[name] for name in names},
                 scene_map_sha256=digest(mapping_content))
    raw = images(a.images)
    if set(raw) != set(ims):
        raise ValueError('Image/COCO mismatch')
    if a.output.exists():
        raise ValueError('Refuse to overwrite existing HRSID preparation')
    from PIL import Image
    import cv2
    split_of = {image: split for split, ids in parts.items() for image in ids}
    gt = {k: [] for k in parts}
    lines = {i: [] for i in ims}
    for n, ann in enumerate(anns):
        if ann['obb'] is None:
            continue
        image = ann['image_id']
        x, y, w, h, theta = ann['obb']
        points = cv2.boxPoints(((x, y), (w, h), math.degrees(theta))).reshape(-1)
        lines[image].append(' '.join(map(str, points.tolist())) + ' ship 0')
        rec = {'image_id': image, 'scene_id': mapping[image], 'class': 'ship', 'obb': ann['obb'], 'gt_id': image + ':' + str(n), 'source_annotation_id': ann['source_annotation_id'], 'source_document': ann['source_document']}
        if shore:
            rec['shore_group'] = shore[image]
        gt[split_of[image]].append(rec)
    splits = {}
    for split, ids in parts.items():
        image_dir = a.output / split / 'images'
        ann_dir = a.output / split / 'annfiles'
        image_dir.mkdir(parents=True)
        ann_dir.mkdir()
        for image in ids:
            with Image.open(io.BytesIO(snapshot(raw[image]))) as im:
                im.convert('RGB').save(image_dir / (image + '.png'))
            (ann_dir / (image + '.txt')).write_text(chr(10).join(lines[image]) + chr(10))
        out = a.output / ('gt_' + split + '.jsonl')
        write_jsonl(out, gt[split])
        counts = audit['counts_by_split'][split]
        if len(gt[split]) != counts['valid_annotations']:
            raise ValueError('HRSID prepared valid count differs from geometry audit')
        splits[split] = {'images': len(ids), 'panoramas': len({mapping[i] for i in ids}),
                         'objects': len(gt[split]), 'objects_raw': counts['source_annotations'],
                         'excluded_objects': counts['excluded_annotations'],
                         'degenerate_exclusions': [row for row in audit['exclusions'] if row['split'] == split],
                         'gt_path': str(out.resolve()), 'gt_sha256': sha256(out)}
    atomic_json(a.output / 'scene_map.json', mapping)
    atomic_json(a.output / 'split.json', {'seed': 0, 'train_fraction': .65, 'parts': parts, 'panorama_counts': no_overlap(parts, mapping)})
    if shore:
        atomic_json(a.output / 'shore_labels.json', {'provenance': shore_provenance, 'labels': shore})
    atomic_json(a.output / 'geometry_audit.json', audit)
    prepared = index_prepared_outputs(a.output, parts, splits)
    manifest = dict(prepared, dataset='hrsid', created_utc=utc(), annotation_choice=a.annotation_choice, annotation_count=len(anns), valid_annotation_count=audit['valid_annotations'], excluded_annotation_count=audit['excluded_annotations'], geometry_policy=dict(GEOMETRY_POLICY), geometry_audit_sha256=sha256(a.output / 'geometry_audit.json'), annotation_hashes={str(f.resolve()): ANNOTATION_HASHES[name] for f,name in zip(a.annotations,names)}, archive_sha256=archive_hash, scene_map_sha256=digest(mapping_content), prepared_scene_map_sha256=sha256(a.output / 'scene_map.json'), split_sha256=sha256(a.output / 'split.json'), official_split_reused=False, shore_labels=shore_provenance, splits=splits)
    atomic_json(a.output / 'data_manifest.json', manifest)
    verify_staged_data(a.output / 'data_manifest.json', manifest)
    atomic_json(marker, {'manifest': str((a.output / 'data_manifest.json').resolve()), 'sha256': sha256(a.output / 'data_manifest.json')})

if __name__ == '__main__':
    main()

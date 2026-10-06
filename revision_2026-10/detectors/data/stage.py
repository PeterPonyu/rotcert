#!/usr/bin/env python3
# Validate already unpacked local data; never downloads, overwrites sources, or deletes.
import argparse
import hashlib
import io
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'export'))
from common import CLASSES, atomic_json, write_jsonl, sha256, scene_id, utc
from canonical import polygon_obb
EXPECTED={'dior':{'trainval':11725,'test':11738},'rsar':{'val':8467,'test':8538},'dv':{'train':17990,'val':1469,'test':8980},'codrone':{'train':5002,'val':2000,'test':3002},'dota':{'val200':5297}}
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
INVENTORY_VERSION=2

def snapshot(path):
    path=Path(path)
    before=path.stat(); content=path.read_bytes(); after=path.stat()
    def identity(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or len(content)!=after.st_size:
        raise ValueError('File changed during snapshot: '+str(path))
    return content

def digest(content):
    return hashlib.sha256(content).hexdigest()

def file_record(path,root,content):
    return {'file':str(path.relative_to(root)),'sha256':digest(content),'bytes':len(content)}

def checked_path(root,name):
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Invalid inventory relative path: '+str(name))
    return Path(root)/rel

def verify_file(root,record):
    path=checked_path(root,record['file']); content=snapshot(path)
    if type(record.get('bytes')) is not int or len(content)!=record['bytes'] or digest(content)!=record.get('sha256'):
        raise ValueError('Staged file changed: '+str(path))
    return content

def decode_image(content,path):
    # Header-only opens miss truncated pixels. Both validators use the hashed bytes.
    from PIL import Image
    import cv2
    import numpy as np
    try:
        with Image.open(io.BytesIO(content)) as im: im.verify()
        with Image.open(io.BytesIO(content)) as im:
            im.load(); width,height=im.size
        decoded=cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or width<=0 or height<=0 or decoded.shape[:2]!=(height,width):
            raise ValueError('Decoder dimensions differ or pixels missing')
    except Exception as error:
        raise ValueError('Undecodable image: '+str(path)) from error
    return width,height

def images(folder):
    files=sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS and p.is_file())
    if not files: raise ValueError('No images: '+str(folder))
    byid={p.stem:p for p in files}
    if len(byid)!=len(files): raise ValueError('Duplicate image stems')
    return byid

def dota_objects(path,dataset,content=None):
    rows=[]
    text=(snapshot(path) if content is None else content).decode('utf-8')
    for i,line in enumerate(text.splitlines()):
        if not line.strip() or line.startswith(('imagesource:','gsd:')): continue
        tokens=line.split()
        if len(tokens)<10: raise ValueError('Malformed annotation '+str(path))
        cls=' '.join(tokens[8:-1])
        if cls not in CLASSES[dataset]: raise ValueError('Unknown class '+cls)
        diff=int(tokens[-1])
        if diff>100: continue
        rows.append((str(i),cls,polygon_obb(list(map(float,tokens[:8])))))
    return rows

DV_POLICY = {
    'id': 'dv-native-valid-geometry-v1',
    'official_native_gt_benchmark': False,
    'deviation': 'Exclude finite exactly-zero source geometry in addition to upstream point-only and star exclusions; not the exact official native GT benchmark.',
    'upstream_commit': '6370b9bc27a2f7b4b76129c9182db87e74225d8c',
    'upstream_loader': 'mmrotate/datasets/dronevehicle.py',
    'upstream_loader_sha256': '96bc625de676df6427caaab1363e2a5d43d125bb3b2838b4781dbe0617bbe770',
    'geometry_priority': ['bndbox', 'polygon', 'point'],
    'polygon_vertices': 'Validate all contiguous finite xN/yN pairs; native loader and canonical GT use only vertices 1..4; preserve all source XML fields.',
    'zero_test': 'Exact rational source coordinates: zero box extent or collinear/coincident polygon vertices; no epsilon.',
    'malformed_nonfinite_unknown_or_unrepresentable': 'fail closed',
    'difficulty': 'unchanged; no new ignore regions',
    'images_and_source_xml': 'preserved',
}
DV_REASONS = ('point_only', 'placeholder_class', 'finite_zero_area_geometry')
DV_AUDIT = 'dv_annotation_audit.json'

def dv_coordinates(node, fields, path):
    import numpy as np
    checked_fields = fields
    if node.tag == 'polygon':
        # The pinned native loader reads x1/y1..x4/y4. Real source XML also
        # contains fifth/sixth vertices: validate every pair, preserve the XML,
        # and return only the coordinates the loader actually consumes.
        if len(node) < 8 or len(node) % 2:
            raise ValueError('Malformed DV coordinate fields: '+str(path))
        checked_fields = [k+str(i) for i in range(1, len(node)//2+1) for k in ('x','y')]
    if len(node) != len(checked_fields) or any(len(node.findall(k)) != 1 for k in checked_fields):
        raise ValueError('Malformed DV coordinate fields: '+str(path))
    try:
        texts = [node.find(k).text.strip() for k in checked_fields]
        values = [float(t) for t in texts]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite DV coordinates')
        exact = [Fraction(t) for t in texts]
        with np.errstate(over='ignore', invalid='ignore'):
            finite32 = np.isfinite(np.asarray(values, dtype=np.float32)).all()
        if not finite32: raise ValueError('Nonfinite DV float32 coordinates')
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('Malformed/nonfinite DV coordinates: '+str(path)) from error
    return exact[:len(fields)]

def prepare_dv_xml(path, content):
    # Native IDs stay in XML; original object positions stay in the inventory.
    tree = ET.fromstring(content)
    if tree.tag != 'annotation': raise ValueError('Malformed DV XML root: '+str(path))
    aliases = {'feright car':'freight car', 'feright_car':'freight car',
               'feright':'freight car', 'truvk':'truck'}
    fields = {'bndbox':['xmin','ymin','xmax','ymax'],
              'polygon':[k+str(j) for j in range(1,5) for k in ('x','y')],
              'point':['x','y']}
    objects = tree.findall('object'); rows = []; excluded = []
    for index, obj in enumerate(objects):
        if len(obj.findall('name')) != 1:
            raise ValueError('Malformed DV class field: '+str(path))
        raw_class = obj.findtext('name') or ''
        cls = aliases.get(raw_class.lower().strip(), raw_class.lower().strip())
        if cls != '*' and cls not in CLASSES['dv']:
            raise ValueError('Unknown class '+cls)
        geometries = {}
        for kind, required in fields.items():
            nodes = obj.findall(kind)
            if len(nodes) > 1: raise ValueError('Malformed duplicate DV geometry: '+str(path))
            if nodes: geometries[kind] = dv_coordinates(nodes[0], required, path)
        if not geometries: raise ValueError('Missing oriented coordinates without point: '+str(path))
        kind = next(k for k in DV_POLICY['geometry_priority'] if k in geometries)
        coords = geometries[kind]
        if kind == 'bndbox':
            xmin,ymin,xmax,ymax = coords
            if xmax < xmin or ymax < ymin: raise ValueError('Malformed inverted DV bndbox: '+str(path))
            coords = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]
        reason = 'placeholder_class' if cls == '*' else ('point_only' if kind == 'point' else None)
        if reason is None:
            points = list(zip(coords[::2], coords[1::2])); x,y = points[0]
            other = next((p for p in points if p != (x,y)), None)
            zero = other is None or all((other[0]-x)*(py-y) == (other[1]-y)*(px-x) for px,py in points)
            if zero: reason = 'finite_zero_area_geometry'
        if reason is not None:
            excluded.append({'object_index':index, 'class_raw':raw_class, 'class':cls,
                'class_id':CLASSES['dv'].index(cls) if cls != '*' else None,
                'native_object_ids':[n.text for n in obj.findall('id')],
                'geometry':kind, 'reason':reason,
                'source_object_xml':ET.tostring(obj, encoding='unicode')})
            tree.remove(obj)
        else:
            # Positive geometry that collapses in float32 must fail, not be excluded.
            rows.append((str(index), cls, polygon_obb([float(v) for v in coords])))
    prepared = ET.tostring(tree, encoding='utf-8') if excluded else content
    return prepared, rows, excluded, len(objects)

def dv_exclusion_records(split, image, source_record, excluded):
    base,mode = split.split('-')
    return [dict(row, id=split+'/'+image+':'+str(row['object_index']),
        split=base, modality=mode, image_id=image, gt_id=image+':'+str(row['object_index']),
        source_xml=source_record['file'], source_xml_sha256=source_record['sha256']) for row in excluded]

def dv_counts(raw, valid, excluded):
    counts = {reason:sum(r['reason']==reason for r in excluded) for reason in DV_REASONS}
    if raw != valid + sum(counts.values()): raise ValueError('DV annotation count conservation failed')
    return dict(counts, raw=raw, valid=valid)

def xml_objects(path,dataset,content=None):
    if dataset=='dv': return prepare_dv_xml(path,snapshot(path) if content is None else content)[1]
    rows=[]; aliases={'feright car':'freight car','feright_car':'freight car','feright':'freight car','truvk':'truck'}
    tree=ET.fromstring(snapshot(path) if content is None else content)
    for i,obj in enumerate(tree.findall('object')):
        cls=obj.findtext('name','').lower().strip(); cls=aliases.get(cls,cls)
        if cls not in CLASSES[dataset]: raise ValueError('Unknown class '+cls)
        if dataset=='dior':
            node=obj.find('robndbox')
            fields=['x_left_top','y_left_top','x_right_top','y_right_top','x_right_bottom','y_right_bottom','x_left_bottom','y_left_bottom']
        else:
            node=obj.find('bndbox')
            if node is not None: fields=['xmin','ymin','xmax','ymin','xmax','ymax','xmin','ymax']
            else: node=obj.find('polygon'); fields=[k+str(j) for j in range(1,5) for k in ('x','y')]
        if node is None: raise ValueError('Missing oriented coordinates: '+str(path))
        coords=[float(node.findtext(k)) for k in fields]
        try:
            box=polygon_obb(coords)
        except ValueError:
            # Match orchestration/prepare_dior_gt.py's disclosed degenerate-GT
            # exclusion. Never alter native XML or invent positive dimensions.
            # Only finite DIOR polygons with an exactly zero fitted side qualify.
            import cv2
            import numpy as np
            points=np.asarray(coords,dtype=np.float32).reshape(4,2)
            if dataset!='dior' or not np.isfinite(points).all(): raise
            sides=cv2.minAreaRect(points)[1]
            if min(sides)!=0: raise
            box=None
        rows.append((str(i),cls,box))
    return rows

def no_overlap(parts,mapping):
    source_sets={}
    for split,ids in parts.items():
        values=[]
        for image in ids:
            source=mapping.get(image)
            if not isinstance(source,str) or not source: raise ValueError('Missing source for '+image)
            values.append(source)
        source_sets[split]=set(values)
    keys=list(parts)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if source_sets[a]&source_sets[b]: raise ValueError('Source leakage between '+a+' and '+b)
    return {k:len(v) for k,v in source_sets.items()}

def verify_directories(root,records):
    for rec in records:
        folder=checked_path(root,rec['directory'])
        actual={p.name for p in folder.iterdir() if p.is_file() and (p.suffix.lower() in EXTS if rec['kind']=='images' else p.suffix==rec['suffix'])}
        expected=rec['names']
        if len(expected)!=len(set(expected)) or actual!=set(expected):
            raise ValueError('Staged directory inventory changed: '+str(folder))

def dv_archive_record(path):
    # Archives can be many GB: hash incrementally, with the snapshot identity gate.
    before=path.stat(); value=sha256(path); after=path.stat()
    def identity(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after): raise ValueError('DV archive changed during hashing: '+str(path))
    return {'sha256':value,'bytes':after.st_size}

def dv_gt_rows(image, objects):
    return [{'image_id':image,'scene_id':scene_id(image,'dv'),'class':cls,
             'obb':box,'gt_id':image+':'+gid} for gid,cls,box in objects]

def dv_audit(source, manifests, excluded):
    counts={split:info['annotation_counts'] for split,info in manifests.items()}
    return {'version':1,'dataset':'dv','source_root':str(source),'policy':DV_POLICY,
            'splits':counts,
            'totals':{key:sum(row[key] for row in counts.values()) for key in ('raw','valid')+DV_REASONS},
            'excluded_objects':excluded}

def verify_dv_data(root, metadata, inventories):
    binding=metadata.get('dv_annotations',{})
    if binding.get('policy')!=DV_POLICY or binding.get('audit_file')!=DV_AUDIT:
        raise ValueError('Re-stage DV: prepared annotation protocol binding required')
    source=Path(binding['source_root'])
    if not source.is_absolute() or source.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(source.resolve()):
        raise ValueError('DV raw/prepared roots must be disjoint')
    required={s+'-'+m for s in EXPECTED['dv'] for m in ('rgb','ir')}
    if set(inventories)!=required: raise ValueError('Incomplete DV split/modality inventory')
    if [r['file'] for r in metadata['controls']] != [DV_AUDIT]:
        raise ValueError('DV excluded-object audit control is mandatory')
    audit_content=verify_file(root,metadata['controls'][0])
    if not metadata.get('archives'): raise ValueError('DV annotation archive binding required')

    expected_directories=[]; exclusions=[]
    for split in sorted(required):
        inv=inventories[split]; info=metadata['splits'][split]
        base,mode=split.split('-'); suffix='r' if mode=='ir' else ''
        image_dir=base+'/'+base+'img'+suffix; ann_dir=base+'/'+base+'label'+suffix
        if (root/base).is_symlink() or (root/ann_dir).is_symlink():
            raise ValueError('DV prepared annotation directories must not alias raw XML')
        if (root/image_dir).resolve()!=(source/image_dir).resolve():
            raise ValueError('DV raw image link changed')
        image_ids=[r['id'] for r in inv['images']]
        if image_ids!=sorted(image_ids) or info['images']!=EXPECTED['dv'][base]:
            raise ValueError('DV image order/count changed')
        if image_ids!=[r['id'] for r in inventories[base+'-'+('ir' if mode=='rgb' else 'rgb')]['images']]:
            raise ValueError('Unpaired DV modalities')
        source_annotations=inv.get('source_annotations',[])
        if image_ids!=[r['id'] for r in source_annotations] or image_ids!=[r['id'] for r in inv['annotations']]:
            raise ValueError('DV raw/prepared annotation mapping changed')
        expected_gt=[]; split_excluded=[]; raw_count=0
        for img,src,prepared in zip(inv['images'],source_annotations,inv['annotations']):
            image=img['id']; expected_ann=ann_dir+'/'+image+'.xml'
            if img['file']!=image_dir+'/'+image+'.jpg' or src['file']!=expected_ann or prepared['file']!=expected_ann:
                raise ValueError('DV loader path binding changed')
            if (img['width'],img['height'])!=(840,712): raise ValueError('DV raw dimensions changed')
            prepared_path=root/prepared['file']
            if prepared_path.is_symlink() or prepared_path.samefile(source/src['file']):
                raise ValueError('DV prepared XML must not alias raw XML')
            content=verify_file(source,src)
            expected,objects,removed,count=prepare_dv_xml(source/src['file'],content)
            actual=verify_file(root,prepared)
            if actual!=expected or prepared.get('source_object_indices')!=[int(row[0]) for row in objects]:
                raise ValueError('DV prepared XML or original object IDs changed')
            expected_gt.extend(dv_gt_rows(image,objects)); raw_count+=count
            split_excluded.extend(dv_exclusion_records(split,image,src,removed))
        gt_content=snapshot(root/('gt_'+split+'.jsonl'))
        observed=[json.loads(line) for line in gt_content.decode('utf-8').splitlines() if line.strip()]
        counts=dv_counts(raw_count,len(expected_gt),split_excluded)
        if (observed!=expected_gt or digest(gt_content)!=info['gt_sha256'] or
                info.get('annotation_counts')!=counts or info['objects']!=counts['valid'] or info['objects_raw']!=counts['raw'] or
                info.get('degenerate_exclusions')!=[r for r in split_excluded if r['reason']=='finite_zero_area_geometry']):
            raise ValueError('DV GT/counts/exclusions do not match loader annotations')
        exclusions.extend(split_excluded)
        for kind,folder,ext in [('images',image_dir,'.jpg'),('annotations',ann_dir,'.xml')]:
            expected_directories.append({'directory':folder,'kind':kind,'suffix':ext,'names':[i+ext for i in image_ids]})
    order=lambda rec:rec['directory']
    if sorted(metadata['directories'],key=order)!=sorted(expected_directories,key=order):
        raise ValueError('DV mandatory directory controls changed')
    verify_directories(source,expected_directories)
    # Ordering is stable independently of JSON key insertion order.
    expected_audit=dv_audit(source,metadata['splits'],sorted(exclusions,key=lambda r:r['id']))
    if json.loads(audit_content)!=expected_audit:
        raise ValueError('DV excluded-object audit does not match source/prepared annotations')
    # Catch changes made while parsing, before a marker or boundary check succeeds.
    for split,inv in inventories.items():
        for row in inv['source_annotations']: verify_file(source,row)
        for row in inv['images']+inv['annotations']: verify_file(root,row)
        if digest(snapshot(root/('gt_'+split+'.jsonl')))!=metadata['splits'][split]['gt_sha256']:
            raise ValueError('DV ground truth changed during verification')
        if digest(snapshot(root/('inventory_'+split+'.json')))!=metadata['splits'][split]['inventory_sha256']:
            raise ValueError('DV inventory changed during verification')
    for name,record in metadata['archives'].items():
        path=Path(name)
        if not path.is_absolute() or dv_archive_record(path)!=record:
            raise ValueError('DV annotation archive changed: '+str(path))
    verify_directories(source,expected_directories)
    verify_file(root,metadata['controls'][0])
    if json.loads(snapshot(root/'data_manifest.json'))!=metadata:
        raise ValueError('DV manifest changed during verification')

def verify_staged_data(manifest_path,metadata):
    """Rehash native staged inputs through the paths the loader actually uses.

    Legacy manifests require re-staging, including HRSID. Phase-boundary checks
    do not make mutable data immutable during a running GPU kernel.
    """
    root=Path(manifest_path).parent
    if metadata.get('dataset')=='codrone':
        raise ValueError('CODrone BLOCKED: complete reviewed frame-to-tile evidence is unavailable')
    if metadata.get('inventory_version')!=INVENTORY_VERSION:
        raise ValueError('Re-stage native data: content inventory v2 required')
    if Path(metadata['root']).resolve()!=root.resolve(): raise ValueError('Staged root mismatch')
    if not metadata.get('splits') or not metadata.get('directories'):
        raise ValueError('Missing staged inventory')
    verify_directories(root,metadata['directories'])
    for record in metadata['controls']: verify_file(root,record)
    inventories={}
    for split,info in metadata['splits'].items():
        path=checked_path(root,'inventory_'+split+'.json'); content=snapshot(path)
        if digest(content)!=info['inventory_sha256']: raise ValueError('Inventory changed: '+str(path))
        inv=json.loads(content)
        if inv.get('version')!=INVENTORY_VERSION: raise ValueError('Re-stage: old split inventory')
        image_ids=[row['id'] for row in inv['images']]
        ann_ids=[row['id'] for row in inv['annotations']]
        if (type(info['images']) is not int or not image_ids or len(image_ids)!=info['images'] or
                len(set(image_ids))!=len(image_ids) or len(set(ann_ids))!=len(ann_ids) or set(ann_ids)!=set(image_ids)):
            raise ValueError('Image/annotation inventory count mismatch')
        for row in inv['images']+inv['annotations']: verify_file(root,row)
        if sum(row['bytes'] for row in inv['images'])!=info['image_bytes']:
            raise ValueError('Image byte count mismatch')
        gt_path=root/('gt_'+split+'.jsonl')
        if Path(info['gt_path']).resolve()!=gt_path.resolve() or digest(snapshot(gt_path))!=info['gt_sha256']:
            raise ValueError('Ground truth changed after staging')
        inventories[split]=inv
    if metadata.get('dataset')=='dv': verify_dv_data(root,metadata,inventories)
    verify_directories(root,metadata['directories'])
    return inventories

def stage(dataset,source,output,state_dir,archives,tiling_manifest=None):
    source=Path(source).resolve(); output=Path(output).absolute(); state_dir=Path(state_dir)
    marker=state_dir/'data_markers'/('DATA_OK_'+dataset); marker.unlink(missing_ok=True)
    if not archives: raise ValueError('At least one complete annotation archive required for provenance')
    hashes={}
    for p in map(Path,archives):
        if not p.is_file() or p.stat().st_size==0 or p.suffix in ('.part','.partial'): raise ValueError('Missing/partial archive '+str(p))
        hashes[str(p.resolve())]=dv_archive_record(p) if dataset=='dv' else {'sha256':sha256(p),'bytes':p.stat().st_size}
    if output.exists() or output.is_symlink() or output==source: raise ValueError('Use a fresh output separate from source data')
    if dataset=='codrone':
        # Do not define an unaudited tiling protocol merely to accept old claims.
        raise ValueError('CODrone BLOCKED: original_counts/tile_size/recipe hash alone do not prove complete tiling; reviewed frame-to-tile inventory and recipe verification required')
    if dataset=='dv' and (output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve())):
        raise ValueError('DV raw/prepared roots must be disjoint')
    parts={}; annotations={}; manifests={}; controls=[]
    if dataset=='dior':
        for split in ('trainval','test'):
            parts[split]=images(source/('JPEGImages-'+split))
            lists=['train','val'] if split=='trainval' else ['test']; ids=[]
            for name in lists:
                path=source/'ImageSets/Main'/(name+'.txt'); content=snapshot(path)
                controls.append(file_record(path,source,content))
                ids.extend(line.strip() for line in content.decode('utf-8').splitlines() if line.strip())
            if len(ids)!=len(set(ids)) or set(ids)!=set(parts[split]): raise ValueError('DIOR split list mismatch')
        if set(parts['trainval'])&set(parts['test']): raise ValueError('DIOR train/test overlap')
        annroot=source/'Annotations/Oriented Bounding Boxes'
        if len(list(annroot.glob('*.xml')))!=sum(EXPECTED['dior'].values()): raise ValueError('DIOR XML count mismatch')
        for split,ims in parts.items(): annotations[split]={i:annroot/(i+'.xml') for i in ims}
    elif dataset=='dv':
        for split in ('train','val','test'):
            for mode,folder,ann in [('rgb',split+'img',split+'label'),('ir',split+'imgr',split+'labelr')]:
                key=split+'-'+mode; parts[key]=images(source/split/folder); annotations[key]={i:source/split/ann/(i+'.xml') for i in parts[key]}
            if set(parts[split+'-rgb'])!=set(parts[split+'-ir']): raise ValueError('Unpaired DV modalities')
    else:
        splits=['train','val0','val200','val500'] if dataset=='dota' else list(EXPECTED[dataset])
        for split in splits:
            parts[split]=images(source/split/'images'); annotations[split]={i:source/split/'annfiles'/(i+'.txt') for i in parts[split]}
        if dataset=='dota':
            no_overlap({'train':parts['train'],'eval':parts['val200']},{i:scene_id(i,'dota') for split in ['train','val200'] for i in parts[split]})
            sources={split:{scene_id(i,'dota') for i in parts[split]} for split in ['val0','val200','val500']}
            if sources['val0']!=sources['val200'] or sources['val500']!=sources['val200']: raise ValueError('DOTA tilings do not cover the same source scenes')
    directories={}
    for split,ims in parts.items():
        target=EXPECTED[dataset].get(split.split('-')[0])
        if target is not None and len(ims)!=target: raise ValueError(f'{dataset}/{split} count {len(ims)} != {target}')
        suffix='.jpg' if dataset in ('dior','dv') else '.png'
        # Pinned RSARDataset globs actual file paths, including mixed JPG/PNG.
        if dataset!='rsar' and any(p.suffix!=suffix for p in ims.values()): raise ValueError('Pinned loader requires '+suffix+' images')
        for kind,paths in [('images',ims),('annotations',annotations[split])]:
            for path in paths.values():
                folder=str(path.parent.relative_to(source)); key=(folder,kind)
                rec=directories.setdefault(key,{'directory':folder,'kind':kind,'suffix':path.suffix,'names':[]})
                rec['names'].append(path.name)
    directories=list(directories.values())
    for rec in directories: rec['names'].sort()
    verify_directories(source,directories)
    output.mkdir(parents=True)
    if dataset=='dv':
        for rec in directories:
            path=output/rec['directory']; path.parent.mkdir(parents=True,exist_ok=True)
            if rec['kind']=='images': path.symlink_to(source/rec['directory'],target_is_directory=True)
            else: path.mkdir()
    else:
        for child in source.iterdir():
            if child.name.startswith(('gt_','inventory_')) or child.name in ('data_manifest.json','scene_map.json','split.json'): continue
            (output/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
    dv_excluded=[]
    for split,ims in parts.items():
        gt=[]; ann_inventory=[]; image_inventory=[]; excluded=[]; source_inventory=[]; objects_raw=0
        for image,path in ims.items():
            content=snapshot(path); width,height=decode_image(content,path)
            if dataset=='dv' and (width,height)!=(840,712): raise ValueError('DV raw image must be 840x712; avoid double cropping')
            image_inventory.append(dict(file_record(path,source,content),id=image,width=width,height=height))
            ann=annotations[split][image]
            if not ann.is_file(): raise ValueError('Missing annotation '+str(ann))
            content=snapshot(ann)
            if dataset=='dv':
                src=dict(file_record(ann,source,content),id=image); source_inventory.append(src)
                prepared,objects,removed,count=prepare_dv_xml(ann,content); objects_raw+=count
                prepared_path=output/ann.relative_to(source)
                with prepared_path.open('xb') as handle: handle.write(prepared)
                ann_inventory.append(dict(file_record(prepared_path,output,prepared),id=image,
                                          source_object_indices=[int(row[0]) for row in objects]))
                excluded.extend(dv_exclusion_records(split,image,src,removed))
            else:
                ann_inventory.append(dict(file_record(ann,source,content),id=image))
                objects=xml_objects(ann,dataset,content) if dataset=='dior' else dota_objects(ann,dataset,content)
            for gid,cls,box in objects:
                if box is None:
                    excluded.append({'image_id':image,'gt_id':image+':'+gid,'class':cls,'reason':'finite_zero_area_polygon'})
                    continue
                gt.append({'image_id':image,'scene_id':scene_id(image,dataset),'class':cls,'obb':box,'gt_id':image+':'+gid})
        gt_path=output/('gt_'+split+'.jsonl'); write_jsonl(gt_path,gt)
        inventory={'version':INVENTORY_VERSION,'images':image_inventory,'annotations':ann_inventory}
        if dataset=='dv': inventory['source_annotations']=source_inventory
        inventory_path=output/('inventory_'+split+'.json'); atomic_json(inventory_path,inventory)
        manifests[split]={'images':len(ims),'objects':len(gt),'objects_raw':len(gt)+len(excluded),'degenerate_exclusions':excluded,'image_bytes':sum(row['bytes'] for row in image_inventory),'inventory_sha256':sha256(inventory_path),'gt_sha256':sha256(gt_path),'gt_path':str(gt_path)}
        if dataset=='dv':
            manifests[split]['annotation_counts']=dv_counts(objects_raw,len(gt),excluded)
            manifests[split]['degenerate_exclusions']=[r for r in excluded if r['reason']=='finite_zero_area_geometry']
            dv_excluded.extend(excluded)
    if dataset=='dv':
        audit_path=output/DV_AUDIT
        atomic_json(audit_path,dv_audit(source,manifests,sorted(dv_excluded,key=lambda r:r['id'])))
        controls.append(file_record(audit_path,output,snapshot(audit_path)))
    manifest={'inventory_version':INVENTORY_VERSION,'dataset':dataset,'root':str(output),'created_utc':utc(),'splits':manifests,'directories':directories,'controls':controls,'archives':hashes,'dv_crop_policy':'runtime CenterCrop only; GT and exported dets raw coordinates' if dataset=='dv' else None,'tiling_manifest_sha256':None}
    if dataset=='dv': manifest['dv_annotations']={'source_root':str(source),'policy':DV_POLICY,'audit_file':DV_AUDIT}
    manifest_path=output/'data_manifest.json'; atomic_json(manifest_path,manifest)
    # Recheck content and exact directory membership before DATA_OK is issued.
    verify_staged_data(manifest_path,manifest)
    atomic_json(marker,{'manifest':str(manifest_path),'sha256':sha256(manifest_path)})
    return manifest

def main():
    p=argparse.ArgumentParser(description='Decode/count/content-hash gates for unpacked native datasets; never downloads')
    p.add_argument('--dataset',choices=list(EXPECTED),required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-dir',type=Path,required=True); p.add_argument('--annotation-archive',action='append',type=Path,required=True); p.add_argument('--tiling-manifest',type=Path)
    a=p.parse_args(); stage(a.dataset,a.source,a.output,a.state_dir,a.annotation_archive,a.tiling_manifest)
if __name__=='__main__': main()

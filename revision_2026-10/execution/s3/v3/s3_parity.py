"""Pinned source comparisons; fixtures/sample IDs fixed before v2 outcomes."""
import ast
import math
import numpy as np
from s3_scores import official_module,author_corners,author_margin,canonicalize_le90
from s3_native import expand_batch,contains_batch,native_readout
from s3_io import ROOT
from rotcert.matching import rotated_iou


def parity(cache,static,cell):
    n=len(cache['tp_scene']);ix=np.linspace(0,n-1,min(32,n),dtype=int)
    p,g=cache['tp_pred'][ix],cache['tp_gt'][ix];official=official_module()
    pp=author_corners(p);gg=author_corners(g)
    oc=np.array([official.convert_to_corners(b) for b in p]);og=np.array([official.convert_to_corners(b) for b in g])
    np.testing.assert_array_equal(pp,oc);np.testing.assert_array_equal(gg,og)
    expected=np.array([official.calculate_expansion_margin(a,b) for a,b in zip(oc,og)])
    np.testing.assert_array_equal(static['expansion_margin'][ix,0],expected)
    # Boundaries include q=0, true margin, neighboring representable thresholds.
    checks=0
    for q in [np.zeros(len(ix)),expected,np.nextafter(expected,np.inf),np.maximum(0,np.nextafter(expected,-np.inf))]:
        expanded=expand_batch(pp,q)
        ref=np.array([official.expand_obb(a,float(v)) for a,v in zip(oc,q)])
        np.testing.assert_array_equal(expanded,ref)
        native=contains_batch(gg,expanded)
        ref_event=np.array([official.is_obb_inside_obb(b,a) for b,a in zip(og,ref)])
        np.testing.assert_array_equal(native,ref_event);checks+=len(q)
    iou=np.array([rotated_iou(a,b) for a,b in zip(p,g)])
    if np.max(abs(iou-cache['tp_iou'][ix]))>2e-10:raise AssertionError('cached IoU differs')
    if np.any(cache['tp_conf']<.05) or np.any(cache['tp_iou']<.5):raise AssertionError('TP population')
    for name,v in static.items():
        if not np.isfinite(v).all() or np.any(v<0):raise AssertionError('invalid score '+name)
    tree=ast.parse((ROOT/'snapshot/run_efficiency_reference.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='coordinate_arrays')
    scope={'np':np,'math':math,'canonicalize_le90':canonicalize_le90}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-coordinate-reference>','exec'),scope)
    for name,v in scope['coordinate_arrays'](p,g).items():np.testing.assert_array_equal(static[name][ix],v)
    return dict(cell_id=cell['cell_id'],fixed_sample_indexes=ix.tolist(),strict_score_exact=True,corner_exact=True,
                native_expansion_exact=True,native_event_exact=True,native_fixture_events=checks,
                coordinate_exact=True,EAV_scope='common matching/class scene-HCP; native event only, not native detector/pose/Hungarian system')

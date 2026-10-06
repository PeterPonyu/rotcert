"""Lazy import alias for torchvision.datapoints (torchvision 0.15) on torchvision >= 0.16, where the module became
torchvision.tv_tensors and BoundingBox became BoundingBoxes. EAV-DETR's COCO transforms reference these names when
src.data.transforms is imported; the CODrone path (CODroneDetection, no transforms, default collate) never calls them.
Installed only in the RotCert EAV venv through a .pth file; the official EAV-DETR source is not modified. The finder is
appended last, so a real torchvision.datapoints would always win."""
import importlib.abc
import importlib.machinery
import sys
import types


class _DatapointsAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torchvision.datapoints':
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        from torchvision import tv_tensors as tv
        m = types.ModuleType(spec.name)
        m.Image, m.Video, m.Mask = tv.Image, tv.Video, tv.Mask
        m.BoundingBox, m.BoundingBoxFormat = tv.BoundingBoxes, tv.BoundingBoxFormat
        m.__rotcert_alias__ = 'torchvision.tv_tensors'
        return m

    def exec_module(self, module):
        return None


sys.meta_path.append(_DatapointsAlias())

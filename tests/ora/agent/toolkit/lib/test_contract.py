"""Versioned read-only vocabulary and data-access boundary checks."""
import ast
import hashlib
import inspect
from pathlib import Path

from ora.agent.toolkit.lib import VERSION, blobs, compilation, control, fusion, geometry, grasp, grasp_review, manifest, motion, observation, perception, pointcloud, vision, pose, clearance, images, numeric, point_images
from ora.agent.toolkit.lib.motion_plan import checks, execute, kinematics, move, orientation, paths, planner, route, timing
from ora.agent.toolkit.interface.types import canonical

ROOT=Path(__file__).resolve().parents[5]/'ora/agent/toolkit/lib'


def _library_files():
    """Flat modules, the manual and every sub-package module (e.g. motion_plan/), by lib-relative path."""
    found=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name!='manifest.py' and (p.suffix=='.py' or p.name=='README.md')]
    return {f'lib/{p.relative_to(ROOT).as_posix()}':hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(found, key=lambda p: p.relative_to(ROOT).as_posix())}


def test_library_manifest_binds_code_and_manual():
    files=_library_files()
    assert manifest.VERSION==VERSION
    assert dict(manifest.FILES)==files
    assert manifest.CONTENT_SHA256==hashlib.sha256(canonical(dict(version=VERSION,files=files))).hexdigest()


def test_lib_has_no_outside_io_or_upper_layer_imports():
    allowed={'hashlib','json','__future__','typing','dataclasses','math','numpy','scipy','io','PIL','cv2'}
    forbidden_calls={'open','read_bytes','read_text','write_bytes','write_text','register','reference',
                     'getenv','eval','exec','__import__'}
    for path in ROOT.rglob('*.py'):
        depth=len(path.relative_to(ROOT).parts)  # 1 for flat modules, 2 inside a sub-package
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert all(a.name.split('.')[0] in allowed for a in node.names),path
            if isinstance(node,ast.ImportFrom):
                # Inside lib (relative levels up to the lib root) or into interface (one more level).
                assert ((1<=node.level<=depth)
                        or (node.level==depth+1 and (node.module or '').startswith('interface.'))
                        or (node.level==0 and node.module.split('.')[0] in allowed)),path
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                if (name=='open' and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                        and node.func.value.id=='Image' and node.args and isinstance(node.args[0],ast.Call)
                        and isinstance(node.args[0].func,ast.Name) and node.args[0].func.id=='BytesIO'):
                    continue  # PIL decoding already granted in-memory bytes, never a filename.
                assert name not in forbidden_calls,(path,name)


def test_public_functions_explain_replay_or_pure_computation():
    for module in (blobs,compilation,control,fusion,geometry,grasp,grasp_review,motion,observation,perception,pointcloud,vision,pose,clearance,images,numeric,point_images,
                   checks,execute,kinematics,move,orientation,paths,planner,route,timing):
        for name,function in inspect.getmembers(module,inspect.isfunction):
            if name.startswith('_') or function.__module__!=module.__name__:continue
            doc=inspect.getdoc(function) or ''
            assert 'Replay:' in doc or 'Pure computation:' in doc,(module.__name__,name)
            assert 'Example:' in doc,(module.__name__,name)

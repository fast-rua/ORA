"""Grounding's public manifests, images and tool results for 0, 1 and 3 candidates.

Each answer is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers answered it
alike.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.toolkit.interface.services import sam3_adapters
from ora.agent.toolkit.interface.tool_data import ToolData
from ora.agent.toolkit.lib import observation
from ora.agent.toolkit.tools.vision_ground.tool import materialize, run
from ora.agent.toolkit.host.runtime.tool_data import configuration
from tests.support.agent.surface_catalog import surface_of
from ora.orchestrator.toolkit.replay import build_api
from ora.orchestrator.toolkit.selection import select
from tests.support.agent.toolkit_lib import BINDINGS
from tests.support.agent.toolkit_goldens import Goldens

ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT/'tests/data/toolkit-general-pickup'
GOLDEN = Goldens(__file__)


class Grounder:
    backend_id = 'sam3'
    backend_version = 'test'
    def __init__(self, count):
        self.count = count
    def configuration(self):
        return dict(endpoint='http://fake.invalid', test=True)
    def ground(self, *, rgb_u8, text, max_candidates):
        rows = []
        for i in range(min(self.count, max_candidates)):
            mask = np.zeros(rgb_u8.shape[:2], dtype=bool)
            mask[160:240, 240+i*5:340+i*5] = True
            rows.append(dict(rank=i, source_index=i, backend_score=.9-i*.1,
                             bbox_xyxy=[240+i*5,160,340+i*5,240], mask=mask))
        return rows, dict(request_id='fixture', returned_candidate_count=len(rows),
                          _service_identity={'test': True}, latency_ms=dict(roundtrip=1.))


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.mark.parametrize('count', [0, 1, 3])
def test_grounding_manifest_images_and_result(golden, tmp_path, count):
    client = Grounder(count)
    surface = surface_of('robodojo_placement_toolkit_v5')
    host = SimpleNamespace(surface=surface, grounding_backend=client, artifact_root=tmp_path/'new')
    selected, events, snapshot = select(FIXTURE, '0')
    api = build_api(FIXTURE, tmp_path/'evaluation', selected, events, snapshot, sam3_adapters(client), BINDINGS)
    api.tool_data = ToolData(store=api.evaluation.store, configuration=configuration(host), replay=True)
    args = selected['proposal']['payload']['arguments']
    actual = run(api, args)
    golden('result', actual, tmp_path, ROOT)
    output = json.loads(api.tool_data.outputs['grounding_query:' + actual['grounding_ref']])
    golden('files', {name: api.read_blob(ref) for name, ref in output['files'].items()}, tmp_path, ROOT)
    manifest = json.loads(api.read_blob(output['files']['manifest.json']))
    assert len(manifest['candidates']) == count
    targets = {}
    for candidate in manifest['candidates']:
        record = api.tool_artifact(kind='target', key=candidate['target_ref'])
        targets[candidate['target_ref']] = {name: api.read_blob(ref) for name, ref in record['files'].items()}
    golden('targets', targets, tmp_path, ROOT)

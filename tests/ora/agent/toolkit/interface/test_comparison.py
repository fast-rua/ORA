"""Content tolerances are independent of volatile service timing and audit bytes."""
import numpy as np
import pytest

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.comparison import compare, VOLATILE_FIELDS
from ora.agent.toolkit.interface.types import canonical


@pytest.mark.parametrize('method', sorted(VOLATILE_FIELDS))
@pytest.mark.parametrize('case,status', [('equal','equal'), ('volatile','equal'),
                                        ('within','within_tolerance'), ('beyond','different')])
def test_fake_service_content_and_envelope_layers(tmp_path, method, case, status):
    stores = [BlobStore(tmp_path / name, writable=True) for name in ('old','new')]
    def service(store, new):
        delta = (5e-7 if case == 'within' else .01 if case == 'beyond' else 0) if new else 0
        meta = dict(request_id='second' if new and case == 'volatile' else 'first',
                    latency_ms={'roundtrip': 9 if new and case == 'volatile' else 1}, model_score=.8+delta)
        if method == 'grasp_candidates':
            data = dict(arrays={'scores':store.put_array(np.array([.8+delta])),
                                'poses_camera':store.put_array(np.eye(4)[None])}, metadata=meta)
        elif method == 'segment':
            data = dict(mask=store.put_array(np.ones((3,4),dtype=bool)), metadata=meta)
        else:
            data = dict(candidates=[dict(source_index=0,rank=0,backend_score=.8+delta,
                         mask=store.put_array(np.ones((3,4),dtype=bool)))],metadata=meta)
        return canonical(dict(method=method, status='ok', data=data))
    old, new = [service(store, bool(i)) for i, store in enumerate(stores)]
    result = compare(method,old,new,reference_store=stores[0],actual_store=stores[1])
    assert result['status'] == status
    assert result['envelope_equal'] == (case == 'equal')
    assert bool(result['differences']) == (case in ('within','beyond'))
    if case == 'volatile':
        assert result['reference_content_sha256'] == result['actual_content_sha256']
        assert result['reference_sha256'] != result['actual_sha256']


def test_mask_counts_candidate_order_and_pose_errors(tmp_path):
    a,b = [BlobStore(tmp_path / n,writable=True) for n in ('a','b')]
    def value(store, flipped):
        mask = np.ones((2,2),dtype=bool)
        mask[0,0] = not flipped
        return canonical(dict(data=dict(candidates=[dict(source_index=int(flipped),rank=0,mask=store.put_array(mask)),
                                                      dict(source_index=int(not flipped),rank=1,mask=store.put_array(mask))])))
    result=compare('ground',value(a,False),value(b,True),reference_store=a,actual_store=b)
    assert result['status']=='different'
    assert result['candidates'][0]['ordering_changed'] is True
    masks=[d for d in result['differences'] if d['kind']=='mask']
    assert masks[0]['different_pixels']==1 and masks[0]['pixel_iou']==.75
    poses=np.repeat(np.eye(4)[None],2,axis=0); poses[1,0,3]=.1
    def grasp(store,poses):
        return canonical(dict(data=dict(arrays=dict(poses_camera=store.put_array(poses)))))
    result=compare('grasp_candidates',grasp(a,poses),grasp(b,poses[::-1].copy()),reference_store=a,actual_store=b)
    assert result['status']=='different' and result['candidates'][0]['ordering_changed']
    assert result['differences'][0]['max_absolute_difference']==.1
    result=compare('grasp_candidates',grasp(a,poses),grasp(b,poses[:1]),reference_store=a,actual_store=b)
    assert result['status']=='different' and result['candidates'][0]['actual_count']==1

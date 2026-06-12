from sglsim.srt.mem_cache.peer_kv_replicator import PeerKVReplicator


def test_peer_kv_replicator_tracks_decode_backup_with_lag():
    replicator = PeerKVReplicator(page_size=16, peer_backup_lag_tokens=2, dp_size=2)

    class Req:
        def __init__(self, rid: str, origin_input_ids: list[int], output_ids: list[int]):
            self.rid = rid
            self.origin_input_ids = origin_input_ids
            self.output_ids = output_ids

    class Batch:
        def __init__(self, reqs):
            self.reqs = reqs

    req = Req("r1", [1, 2, 3, 4], [])
    replicator.on_prefill_complete(req)
    for tok in [10, 11, 12, 13]:
        req.output_ids.append(tok)
        replicator.on_decode_step(Batch([req]))

    backed_up = replicator.count_peer_backed_up_for_req("r1")
    assert backed_up == len(req.origin_input_ids) + 2

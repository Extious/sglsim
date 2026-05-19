from tools.check_sglang_native_parity import PARITY_FILES, build_manifest


def test_upstream_reference_files_exist():
    manifest = build_manifest()
    missing = [
        rel
        for rel in PARITY_FILES
        if manifest[rel]["upstream_exists"] != "True"
    ]
    assert missing == []

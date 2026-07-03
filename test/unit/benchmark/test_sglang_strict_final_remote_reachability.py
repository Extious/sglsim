from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHECK_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "check_strict_final_remote_reachability.py"
)


def load_check_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_remote_reachability",
        CHECK_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classifies_common_ssh_failures() -> None:
    module = load_check_module()

    assert module.classify_ssh_status(0, "ok") == "ok"
    assert (
        module.classify_ssh_status(
            255,
            "Bad owner or permissions on /etc/ssh/ssh_config.d/05-redhat.conf",
        )
        == "ssh_config_error"
    )
    assert (
        module.classify_ssh_status(
            255,
            "ssh: connect to host 158.182.7.1 port 22: Network is unreachable",
        )
        == "network_unreachable"
    )
    assert (
        module.classify_ssh_status(
            255,
            "ssh: Could not resolve hostname faith.comp.hkbu.edu.hk",
        )
        == "dns_unresolved"
    )


def test_reachability_payload_uses_current_manifest_jump_host() -> None:
    module = load_check_module()
    manifest = {
        "ssh": {
            "alias": "hkbugpumanode",
            "identity_file": "~/.ssh/id_rsa",
            "jump_host": "faith.comp.hkbu.edu.hk",
            "jump_user": "zhanzhao",
            "remote_host": "172.27.244.19",
            "remote_user": "25482351",
            "timeout_s": 20,
        }
    }

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        text = " ".join(command)
        if command[:2] == ["getent", "hosts"] and "faith.comp.hkbu.edu.hk" in text:
            return subprocess.CompletedProcess(command, 0, "1.2.3.4 faith.comp.hkbu.edu.hk\n")
        if command[:2] == ["getent", "ahosts"] and "172.27.244.19" in text:
            return subprocess.CompletedProcess(
                command,
                0,
                "172.27.244.19 STREAM 172.27.244.19\n",
            )
        if "faith.comp.hkbu.edu.hk" in text:
            return subprocess.CompletedProcess(
                command,
                255,
                "ssh: Could not resolve hostname faith.comp.hkbu.edu.hk",
            )
        return subprocess.CompletedProcess(
            command,
            255,
            "Bad owner or permissions on /etc/ssh/ssh_config.d/05-redhat.conf",
        )

    payload = module.run_reachability_checks(manifest, runner=fake_runner)

    assert payload["overall_status"] == "unreachable"
    assert payload["current_generated_direct_proxy_jump_host"] == "faith.comp.hkbu.edu.hk"
    statuses = {item["id"]: item["status"] for item in payload["checks"]}
    assert statuses["alias_default"] == "ssh_config_error"
    assert statuses["direct_proxy_faith_comp_hkbu_edu_hk"] == "dns_unresolved"
    assert statuses["getent_jump_faith_comp_hkbu_edu_hk"] == "ok"
    assert statuses["getent_remote_172_27_244_19"] == "ok"
    assert payload["primary_submission_route"] == "direct-proxy"
    assert payload["submission_blockers"] == [
        "direct_proxy_faith_comp_hkbu_edu_hk=dns_unresolved"
    ]
    assert payload["diagnostic_failures"] == ["alias_default=ssh_config_error"]
    assert "Remote target IP diagnostic: ok" in payload["impact"]
    assert payload["resolver"]["path"] == "/etc/resolv.conf"
    assert any(
        "--jump-host '<reachable_jump_host_or_ip>'" in command
        for command in payload["next_step_commands"]
    )


def test_numeric_remote_host_uses_ahosts() -> None:
    module = load_check_module()

    assert module.getent_command_for_host("172.27.244.19") == [
        "getent",
        "ahosts",
        "172.27.244.19",
    ]
    assert module.getent_command_for_host("faith.comp.hkbu.edu.hk") == [
        "getent",
        "hosts",
        "faith.comp.hkbu.edu.hk",
    ]


def test_direct_proxy_submission_blockers_are_separate_from_diagnostics() -> None:
    module = load_check_module()
    checks = [
        {"id": "alias_default", "status": "ssh_config_error"},
        {"id": "direct_proxy_faith_comp_hkbu_edu_hk", "status": "dns_unresolved"},
        {"id": "getent_remote_172_27_244_19", "status": "ok"},
    ]

    assert module.submission_blockers(checks) == [
        "direct_proxy_faith_comp_hkbu_edu_hk=dns_unresolved"
    ]
    assert module.diagnostic_failures(checks) == ["alias_default=ssh_config_error"]


def test_reads_resolver_config(tmp_path: Path) -> None:
    module = load_check_module()
    resolv = tmp_path / "resolv.conf"
    resolv.write_text(
        "\n".join(
            [
                "# generated",
                "search comp.hkbu.edu.hk",
                "nameserver 158.182.7.12",
                "nameserver 158.182.7.13",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.read_resolver_config(resolv)

    assert payload["exists"] is True
    assert payload["search"] == ["comp.hkbu.edu.hk"]
    assert payload["nameservers"] == ["158.182.7.12", "158.182.7.13"]

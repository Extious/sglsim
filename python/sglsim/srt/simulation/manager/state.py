from __future__ import annotations


class StateManager:
    _iteration: int = 0
    _last_inference_dur: float = 0.0
    _current_inference_dur: float = 0.0
    _hicache_l2_load_dur: float = 0.0
    _hicache_l2_backup_dur: float = 0.0
    _comm_h2d_s: float = 0.0
    _comm_d2h_s: float = 0.0
    _comm_disk_write_s: float = 0.0

    @classmethod
    def reset(cls) -> None:
        cls._iteration = 0
        cls._last_inference_dur = 0.0
        cls._current_inference_dur = 0.0
        cls._hicache_l2_backup_dur = 0.0
        cls._hicache_l2_load_dur = 0.0
        cls._comm_h2d_s = 0.0
        cls._comm_d2h_s = 0.0
        cls._comm_disk_write_s = 0.0

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur

    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0.0
        return dur

    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0.0
        return dur

    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur

    @classmethod
    def set_last_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = dur

    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur

    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur

    @classmethod
    def inc_comm_h2d_s(cls, dur: float) -> None:
        if dur <= 0:
            return
        cls._comm_h2d_s += dur

    @classmethod
    def inc_comm_d2h_s(cls, dur: float) -> None:
        if dur <= 0:
            return
        cls._comm_d2h_s += dur

    @classmethod
    def inc_comm_disk_write_s(cls, dur: float) -> None:
        if dur <= 0:
            return
        cls._comm_disk_write_s += dur

    @classmethod
    def get_comm_totals(cls) -> dict[str, float]:
        return {
            "comm_h2d_s": cls._comm_h2d_s,
            "comm_d2h_s": cls._comm_d2h_s,
            "comm_disk_write_s": cls._comm_disk_write_s,
        }

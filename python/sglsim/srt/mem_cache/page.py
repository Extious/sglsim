from dataclasses import dataclass


@dataclass
class KVPage:
    page_id: int
    page_size: int
    owner: str | None = None

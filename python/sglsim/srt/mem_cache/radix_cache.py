from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Node:
    page: int | None = None
    children: dict[int, "_Node"] = field(default_factory=dict)


class PrefixCache:
    def __init__(self):
        self._root = _Node()

    def insert(self, tokens: list[int], pages: list[int]) -> None:
        if len(tokens) != len(pages):
            raise ValueError("tokens and pages must have the same length")
        node = self._root
        for token, page in zip(tokens, pages):
            node = node.children.setdefault(token, _Node())
            node.page = page

    def match(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        node = self._root
        matched_tokens = []
        matched_pages = []
        for token in tokens:
            child = node.children.get(token)
            if child is None or child.page is None:
                break
            matched_tokens.append(token)
            matched_pages.append(child.page)
            node = child
        return matched_tokens, matched_pages


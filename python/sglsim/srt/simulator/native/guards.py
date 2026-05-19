from __future__ import annotations


class AuthoritativePathViolation(RuntimeError):
    pass


def assert_not_runtime_service_path(component: str) -> None:
    raise AuthoritativePathViolation(
        f"{component} is using RuntimeService; native SGLang parity path must use "
        "TokenizerManager/DataParallelController/Scheduler instead."
    )

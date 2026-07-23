from __future__ import annotations

from typing import Any


def render_prompt(template: str, **kwargs: Any) -> str:
    safe_kwargs = {k: (v if v is not None else "") for k, v in kwargs.items()}
    try:
        return template.format(**safe_kwargs)
    except KeyError as e:
        missing = e.args[0]
        provided = ", ".join(sorted(safe_kwargs.keys()))
        raise KeyError(
            f"Missing prompt variable '{missing}'. Provided keys: [{provided}]"
        ) from e

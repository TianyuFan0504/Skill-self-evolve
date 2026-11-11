"""Register the ALFWorld Task in each worker and delegate to upstream Uni-Agent."""

from __future__ import annotations

import copy

from . import task  # noqa: F401 -- registration must happen in every Ray worker


async def run_task(**kwargs):
    from uni_agent.framework.task_runner import run_task as upstream_run_task

    if kwargs.get("tools_kwargs") is not None:
        tools_kwargs = copy.deepcopy(kwargs["tools_kwargs"])
        sample = tools_kwargs.get("task")
        if isinstance(sample, dict) and kwargs.get("sample_index") is not None:
            sample.setdefault("metadata", {})["sample_index"] = kwargs["sample_index"]
        kwargs["tools_kwargs"] = tools_kwargs
    return await upstream_run_task(**kwargs)

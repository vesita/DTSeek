"""任务卡插件体系：协议 + 运行时 + 引擎 + 探针 + 内置卡。

`import dtseek.tasks` 会把内置卡一并 import 进来并完成注册，`all_tasks()` 随即可用。
"""
from dtseek.tasks import builtin  # noqa: F401

__all__ = ["builtin"]

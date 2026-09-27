"""DTSeek: 自回归切片决策引擎 —— 共享基座 + 可插拔任务卡。"""

from .decoder.robust_ar_model import RobustARSliceDecoder
from .encoder.nano_doc_encoder import NanoDocEncoder
from .encoder.segmenter import split_with_global_offsets
from .tasks.engine import MultiTaskEngine
from .tasks.plugin import TaskCard, TaskSpec, all_tasks

__all__ = [
    "RobustARSliceDecoder",
    "NanoDocEncoder",
    "split_with_global_offsets",
    "MultiTaskEngine",
    "TaskCard",
    "TaskSpec",
    "all_tasks",
]

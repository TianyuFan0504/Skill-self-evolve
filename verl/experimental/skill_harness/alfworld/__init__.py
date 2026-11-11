"""ALFWorld tasks and offline data preparation for the native Uni-Agent harness.

Import ``runner`` in rollout workers to register the training Task.  Keeping this
module lightweight lets the catalog helpers run without Uni-Agent or TextWorld.
"""

TASK_NAME = "alfworld_skill_training"


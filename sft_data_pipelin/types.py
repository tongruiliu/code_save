from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


@dataclass
class Action:
    name: str
    kwargs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    user_id: str
    instruction: str
    actions: List[Action]
    outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "instruction": self.instruction,
            "actions": [a.to_dict() for a in self.actions],
            "outputs": list(self.outputs),
        }


@dataclass
class RewardInfo:
    r_actions: bool
    gt_data_hash: str
    r_outputs: float
    outputs: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvInfo:
    task: Task
    source: Optional[str] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "source": self.source,
            "user_cost": self.user_cost,
            "reward_info": None if self.reward_info is None else self.reward_info.to_dict(),
        }


@dataclass
class EnvResetResponse:
    observation: str
    info: EnvInfo


@dataclass
class EnvResponse:
    observation: str
    reward: float
    done: bool
    info: EnvInfo


@dataclass
class SolveResult:
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    total_cost: float = 0.0


@dataclass
class EnvRunResult:
    task_id: int
    reward: float
    info: Dict[str, Any]
    traj: List[Dict[str, Any]]
    trial: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
    target_image_url: Optional[str] = None
    target_image_path: Optional[str] = None
    target_canvas: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "user_id": self.user_id,
            "instruction": self.instruction,
            "actions": [a.to_dict() for a in self.actions],
            "outputs": list(self.outputs),
        }
        if self.target_image_url:
            payload["target_image_url"] = self.target_image_url
        if self.target_image_path:
            payload["target_image_path"] = self.target_image_path
        if self.target_canvas is not None:
            payload["target_canvas"] = self.target_canvas
        return payload


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
    reward_info: Optional[RewardInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "source": self.source,
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


@dataclass
class EnvRunResult:
    task_id: int
    reward: float
    info: Dict[str, Any]
    traj: List[Dict[str, Any]]
    trial: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

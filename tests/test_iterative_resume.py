"""Tests for no-retry resume behavior."""

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import Mock

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic
from openhands.sdk.workspace import RemoteWorkspace


class MockEvaluation(Evaluation):
    """Mock evaluation class for testing."""

    def __init__(self, *args, instances: List[EvalInstance], **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_test_instances", instances)

    def prepare_instances(self) -> List[EvalInstance]:
        return object.__getattribute__(self, "_test_instances")

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        mock_workspace = Mock(spec=RemoteWorkspace)
        mock_workspace.__enter__ = Mock(return_value=mock_workspace)
        mock_workspace.__exit__ = Mock(return_value=None)
        mock_workspace.forward_env = forward_env or []
        mock_workspace.resource_factor = resource_factor
        return mock_workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        return EvalOutput(
            instance_id=instance.id,
            test_result={"git_patch": "mock patch"},
            instruction="mock instruction",
            error=None,
            history=[],
            instance=instance.data,
        )


def _make_evaluation(tmpdir: str, instances: list[EvalInstance]) -> MockEvaluation:
    metadata = EvalMetadata(
        llm=LLM(model="test-model", temperature=0.0),
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=tmpdir,
        details={},
        eval_limit=len(instances),
        n_critic_runs=3,
        max_retries=3,
        critic=PassCritic(),
    )
    return MockEvaluation(metadata=metadata, num_workers=1, instances=instances)


def _write_output(tmpdir: str, instance_id: str, patch: str) -> None:
    output = EvalOutput(
        instance_id=instance_id,
        test_result={"git_patch": patch},
        instruction="mock",
        error=None,
        history=[],
        instance={"test": instance_id},
    )
    with open(os.path.join(tmpdir, "output.jsonl"), "a", encoding="utf-8") as f:
        f.write(output.model_dump_json() + "\n")


def _write_trajectory(tmpdir: str, instance_id: str, filename: str = "trajectory.json"):
    conversation_dir = Path(tmpdir) / "conversations" / instance_id
    conversation_dir.mkdir(parents=True, exist_ok=True)
    (conversation_dir / filename).write_text("[]\n", encoding="utf-8")


def test_metadata_is_clamped_to_no_retry():
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluation = _make_evaluation(
            tmpdir,
            [EvalInstance(id="A", data={"test": "A"})],
        )

        assert evaluation.metadata.n_critic_runs == 1
        assert evaluation.metadata.max_retries == 0


def test_restart_skip_requires_non_empty_patch_and_trajectory():
    with tempfile.TemporaryDirectory() as tmpdir:
        instances = [
            EvalInstance(id=inst_id, data={"test": inst_id})
            for inst_id in ["A", "B", "C", "D", "E"]
        ]
        evaluation = _make_evaluation(tmpdir, instances)

        _write_output(tmpdir, "A", "diff --git a/a b/a")
        _write_trajectory(tmpdir, "A", "trajectory.jsonl")

        _write_output(tmpdir, "B", "diff --git a/b b/b")

        _write_output(tmpdir, "C", "")
        _write_trajectory(tmpdir, "C", "trajectory.jsonl")

        _write_output(tmpdir, "D", "diff --git a/d b/d")
        _write_trajectory(tmpdir, "D", "trajectory.json")

        result = evaluation._get_instances_for_attempt(
            attempt=1,
            all_instances=instances,
            critic=PassCritic(),
        )

        assert {inst.id for inst in result} == {"B", "C", "D", "E"}


def test_later_attempts_are_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        instances = [EvalInstance(id="A", data={"test": "A"})]
        evaluation = _make_evaluation(tmpdir, instances)

        result = evaluation._get_instances_for_attempt(
            attempt=2,
            all_instances=instances,
            critic=PassCritic(),
        )

        assert result == []


def test_trajectory_writer_creates_json_and_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluation = _make_evaluation(tmpdir, [])
        conversation_dir = Path(tmpdir) / "conversations" / "A"
        events_dir = conversation_dir / "workspace" / "conversations" / "abc" / "events"
        events_dir.mkdir(parents=True)
        (events_dir / "event-0001.json").write_text(
            json.dumps({"id": 1, "tool_name": "terminal"}),
            encoding="utf-8",
        )

        paths = evaluation._write_trajectory_files(conversation_dir)

        assert paths is not None
        json_path, jsonl_path = paths
        assert json_path.name == "trajectory.json"
        assert jsonl_path.name == "trajectory.jsonl"
        assert json_path.stat().st_size > 0
        assert jsonl_path.stat().st_size > 0
        assert json.loads(json_path.read_text(encoding="utf-8")) == [
            {"id": 1, "tool_name": "terminal"}
        ]
        assert json.loads(jsonl_path.read_text(encoding="utf-8")) == {
            "id": 1,
            "tool_name": "terminal",
        }

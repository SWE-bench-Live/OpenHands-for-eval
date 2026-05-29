import json

from benchmarks.utils.evaluation_utils import get_default_on_result_writer
from benchmarks.utils.models import EvalInstance, EvalOutput


def test_result_writer_writes_errors_to_output_and_error_file(tmp_path):
    output_path = tmp_path / "output.jsonl"
    writer = get_default_on_result_writer(str(output_path))
    instance = EvalInstance(id="A", data={})
    output = EvalOutput(
        instance_id="A",
        test_result={"git_patch": "diff --git a/a b/a"},
        instruction="mock",
        error="boom",
        history=[],
        instance={},
    )

    writer(instance, output)

    output_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    error_rows = [
        json.loads(line)
        for line in (tmp_path / "output_errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [row["instance_id"] for row in output_rows] == ["A"]
    assert [row["instance_id"] for row in error_rows] == ["A"]
    assert output_rows[0]["error"] == "boom"
    assert error_rows[0]["test_result"]["git_patch"] == "diff --git a/a b/a"

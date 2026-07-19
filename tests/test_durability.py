import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multistep_llm_pipeline import MultiStepLLMPipeline
from tests.fake_client import FakeLLMClient, RecordingLogger, Return

PASSED, FAILED = [], []

SENSE = "User reports the application crashes on upload."
INTENT = '{"intent": "support", "confidence": 5, "reason": "user reports a crash"}'
FIELDS = json.dumps({
    "summary": "app crashes on upload",
    "sentiment": "negative",
    "urgency": "high",
    "language": "en",
    "intent": "support",
    "problem": "crash on upload",
    "product": "application",
    "error_code": None,
})
ANSWER = "Sorry about that. Could you tell us which file type triggers the crash?"
JUDGE = '{"passed": true, "score": 9, "issues": []}'

LINES = [
    "The application crashes every time I try to upload a file.",
    "I cannot log into my account and need help resetting my password.",
    "My payment was declined and I need assistance resolving the issue.",
]

def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    mark = "  ok  " if condition else " FAIL "
    print(f"[{mark}] {name}" + (f"  -> {detail}" if detail and not condition else ""))

def build_pipeline(workdir, judge_script=None, answer_script=None):
    input_path = os.path.join(workdir, "input.txt")
    output_path = os.path.join(workdir, "output.txt")

    with open(input_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")

    logger = RecordingLogger()
    pipeline = MultiStepLLMPipeline(
        extract_client=FakeLLMClient([Return(SENSE)]),
        classifier_client=FakeLLMClient([Return(INTENT)]),
        field_extractor_client=FakeLLMClient([Return(FIELDS)]),
        response_generator_client=FakeLLMClient(answer_script or [Return(ANSWER)]),
        judge_client=FakeLLMClient(judge_script or [Return(JUDGE)]),
        input_path=input_path,
        output_path=output_path,
        logger=logger,
    )

    for guard in pipeline.guards.values():
        guard.sleep = lambda _: None

    return pipeline, logger, output_path

def read_partial(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def test_happy_path():
    print("\n--- durability: clean run ---")
    workdir = tempfile.mkdtemp()

    try:
        pipeline, logger, output_path = build_pipeline(workdir)
        summary = pipeline.process_generation_pipeline()

        check("all lines completed", summary["lines_completed"] == 3, str(summary["lines_completed"]))
        check("no failed lines", summary["lines_failed"] == [], str(summary["lines_failed"]))
        check("no degraded lines", summary["lines_degraded"] == [], str(summary["lines_degraded"]))

        partial = read_partial(pipeline.partial_path)
        check("partial holds every result", len(partial) == 3, str(len(partial)))

        with open(output_path, encoding="utf-8") as f:
            final = json.load(f)

        check("output matches partial", len(final) == len(partial))
        check("fallback levels recorded", final[0]["fallback_levels"].get("final_answer") == 0,
              str(final[0].get("fallback_levels")))
    finally:
        shutil.rmtree(workdir)

def test_optional_step_degrades():
    print("\n--- criticality: optional judge does not drop the line ---")
    workdir = tempfile.mkdtemp()

    try:
        pipeline, logger, output_path = build_pipeline(workdir, judge_script=[Return("not json ever")])
        summary = pipeline.process_generation_pipeline()

        check("lines still complete", summary["lines_completed"] == 3, str(summary["lines_completed"]))
        check("lines marked degraded", len(summary["lines_degraded"]) == 3, str(summary["lines_degraded"]))

        with open(output_path, encoding="utf-8") as f:
            final = json.load(f)

        check("judge_result is null", final[0]["judge_result"] is None)
        check("degraded_steps names judge", final[0]["degraded_steps"] == ["judge"],
              str(final[0]["degraded_steps"]))
        check("final answer preserved", final[0]["final_answer"] == ANSWER)
    finally:
        shutil.rmtree(workdir)

def test_required_step_drops_line():
    print("\n--- criticality: required final_answer drops the line ---")
    workdir = tempfile.mkdtemp()

    try:
        pipeline, logger, output_path = build_pipeline(workdir, answer_script=[Return("")])
        summary = pipeline.process_generation_pipeline()

        check("no lines completed", summary["lines_completed"] == 0, str(summary["lines_completed"]))
        check("all lines reported failed", summary["lines_failed"] == [0, 1, 2], str(summary["lines_failed"]))
        check("output not written", not os.path.exists(output_path))
        check("partial file is empty", read_partial(pipeline.partial_path) == [])
    finally:
        shutil.rmtree(workdir)

def test_line_isolation():
    print("\n--- durability: one bad line does not kill the run ---")
    workdir = tempfile.mkdtemp()

    try:
        pipeline, logger, output_path = build_pipeline(workdir)
        original = pipeline.generate_pipeline_step

        def exploding(msg, step=-1):
            if step == 1:
                raise RuntimeError("synthetic failure inside line 1")
            return original(msg, step=step)

        pipeline.generate_pipeline_step = exploding
        summary = pipeline.process_generation_pipeline()

        check("surviving lines completed", summary["lines_completed"] == 2, str(summary["lines_completed"]))
        check("crashed line reported", summary["lines_failed"] == [1], str(summary["lines_failed"]))

        with open(output_path, encoding="utf-8") as f:
            final = json.load(f)

        check("output keeps lines 0 and 2", [r["question_index"] for r in final] == [0, 2],
              str([r["question_index"] for r in final]))
        check("error logged", "synthetic failure" in logger.text())
    finally:
        shutil.rmtree(workdir)

def test_hard_crash_preserves_finished_lines():
    print("\n--- durability: hard crash keeps completed work on disk ---")
    workdir = tempfile.mkdtemp()

    try:
        pipeline, logger, output_path = build_pipeline(workdir)
        original = pipeline.generate_pipeline_step

        def exploding(msg, step=-1):
            if step == 2:
                raise KeyboardInterrupt("simulated hard stop")
            return original(msg, step=step)

        pipeline.generate_pipeline_step = exploding
        crashed = False

        try:
            pipeline.process_generation_pipeline()
        except KeyboardInterrupt:
            crashed = True

        check("hard interrupt propagates", crashed)

        partial = read_partial(pipeline.partial_path)
        check("finished lines survived on disk", len(partial) == 2, str(len(partial)))
        check("partial keeps line indexes", [r["question_index"] for r in partial] == [0, 1],
              str([r["question_index"] for r in partial]))
        check("partial rows are complete results", bool(partial[0]["final_answer"]))
    finally:
        shutil.rmtree(workdir)

def main():
    test_happy_path()
    test_optional_step_degrades()
    test_required_step_drops_line()
    test_line_isolation()
    test_hard_crash_preserves_finished_lines()

    print("\n" + "=" * 60)
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    for name in FAILED:
        print(f"  FAIL: {name}")
    print("=" * 60)

    return 1 if FAILED else 0

if __name__ == "__main__":
    sys.exit(main())

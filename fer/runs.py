"""Where runs are recorded, and the list of planned runs the dashboard shows."""
import json
from pathlib import Path

RUNS = Path("runs")


def record_path(stage, model, iteration):
    return RUNS / stage / model / f"{iteration}.json"


def add_to_plan(entries):
    """entries: dicts with at least stage, model, iteration, epochs."""
    path = RUNS / "plan.json"
    plan = json.loads(path.read_text()) if path.exists() else []
    key = lambda e: (e["stage"], e["model"], str(e["iteration"]))
    known = {key(e) for e in plan}
    plan += [e for e in entries if key(e) not in known]
    RUNS.mkdir(exist_ok=True)
    path.write_text(json.dumps(plan, indent=1))


def is_done(stage, model, iteration):
    p = record_path(stage, model, iteration)
    return p.exists() and json.loads(p.read_text()).get("status") == "done"

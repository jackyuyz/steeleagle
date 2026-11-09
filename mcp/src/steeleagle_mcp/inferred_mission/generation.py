from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from steeleagle_mcp.mission_tools import translate_with_dsl_reference_payload
from steeleagle_mcp.nl2dsl.pipeline import run_dsl_through_pipeline

UNBOUND_AREA = "UNBOUND_AREA"
_MAIN_ACTIONS = {
    "track": ("Track", "ReturnToHome"),
    "patrol": ("Patrol", "ReturnToHome"),
    "survey": ("Patrol", "ReturnToHome"),
    "climb": ("ElevateToAltitude", "Wait", "ReturnToHome"),
    "wait": ("Wait", "ReturnToHome"),
    "precision_land": ("PrecisionLand",),
    "land": ("Land",),
}


def _check_mission(mission: dict, task: dict, area: str | None) -> list[str]:
    """Check the task, parameters and transitions survive translation."""
    actions, events, transitions = (mission[key] for key in ("actions", "events", "transitions"))
    chain = list(_MAIN_ACTIONS[task["task"]])
    if "takeoff" in task:
        chain.insert(0, "TakeOff")
    fallback = task["task"] == "precision_land"
    if sorted(a["type_name"] for a in actions.values()) != sorted(chain + (["Land"] if fallback else [])):
        return [f"Expected actions {chain}" + (" with a Land battery fallback." if fallback else ".")]

    errors = []
    action_id = mission["start_action_id"]
    by_type = {a["type_name"]: key for key, a in actions.items()}
    battery_action = _MAIN_ACTIONS[task["task"]][0] if task["task"] in {
        "track", "patrol", "survey", "precision_land"
    } else None
    for action_type in chain:
        if action_id != by_type[action_type]:
            errors.append(f"Expected {action_type} on the done path.")
            break
        edges = transitions.get(action_id, {})
        conditional = {k: v for k, v in edges.items() if k != "done"}
        if action_type == battery_action:
            expected = by_type["Land" if fallback else "ReturnToHome"]
            valid_battery = len(conditional) == 1 and all(
                target == expected and events[event]["type_name"] == "BatteryReached"
                and events[event]["attributes"]["threshold"] == 30
                for event, target in conditional.items()
            )
            if not valid_battery:
                errors.append(f"{action_type} needs exactly one battery threshold 30 transition to its fallback.")
        elif conditional:
            errors.append(f"Unexpected conditional transition on {action_type}.")
        action_id = edges.get("done", "terminate")
    if action_id != "terminate":
        errors.append("The normal completion path must terminate after the final action.")
    if fallback and transitions.get(by_type["Land"], {"done": "terminate"}) != {"done": "terminate"}:
        errors.append("The battery fallback Land must terminate on completion.")

    attributes = {name: actions[key]["attributes"] for name, key in by_type.items()}
    if "TakeOff" in attributes and attributes["TakeOff"].get("take_off_altitude") != task["takeoff"]:
        errors.append("TakeOff altitude must match the inference.")
    if "Wait" in attributes and attributes["Wait"].get("duration") != task["seconds"]:
        errors.append("Wait duration must match the inference.")
    if "ElevateToAltitude" in attributes and attributes["ElevateToAltitude"].get("target_altitude") != task["alt"]:
        errors.append("Climb altitude must match the inference.")
    if "target" in task:
        target_action = "PrecisionLand" if fallback else "Track"
        if attributes[target_action]["target"]["class_name"] != task["target"]:
            errors.append("The detection target class must match the inference.")
    if "Patrol" in attributes:
        plan = attributes["Patrol"]["plan"]
        algo = "survey" if task["task"] == "survey" else "edge"
        if plan.get("area") != (area or UNBOUND_AREA) or plan.get("algo") != algo or plan.get("alt") != task["alt"]:
            errors.append(f"Use RoutePlan area {area or UNBOUND_AREA}, algo {algo}, and alt {task['alt']}.")
    return errors


class Generation:
    """Reference and compile tools for one analysis section, without vehicle actions."""

    def __init__(self, instruction: str, task: dict, area: str | None):
        self.instruction = instruction
        self.task = task
        self.area = area
        self.attempts: list[dict] = []

    def reference(self) -> dict:
        result = translate_with_dsl_reference_payload(self.instruction, include_examples=False)
        if not result["ok"]:
            return result
        allowed = set(_MAIN_ACTIONS[self.task["task"]]) | {"TakeOff", "Land"}
        schema = result["reference"]["schema"]
        schema["actions"] = {key: value for key, value in schema["actions"].items() if key in allowed}
        schema["events"] = {key: value for key, value in schema["events"].items() if key == "BatteryReached"}
        schema["data"] = {key: value for key, value in schema["data"].items() if key in {"Detection", "RoutePlan"}}
        # Keep the prompt reproducible on a clean checkout, without local examples.
        result["reference"]["few_shot_examples"] = []
        result["recommended_workflow"] = ["Write the requested DSL and call compile_mission_dsl. Fix reported errors."]
        rule = (
            f"Use RoutePlan area = {self.area or UNBOUND_AREA} for a route. "
            "UNBOUND_AREA is a visible analysis placeholder, not a real KML area. "
            "Keep the exact requested tasks, parameters and transitions. "
            "A PrecisionLand completion terminates normally, with Land only on the battery transition."
        )
        result["reference"]["generation_rules"] = [*result["reference"]["generation_rules"], rule]
        return result

    def dispatch(self, name: str, args: dict) -> dict:
        if name == "translate_with_dsl_reference":
            return self.reference()
        if name != "compile_mission_dsl":
            return {"ok": False, "errors": ["Only DSL reference and compilation tools are available."]}
        dsl = args.get("dsl")
        if not isinstance(dsl, str) or not dsl.strip() or len(dsl) > 24_000:
            return {"ok": False, "errors": ["Provide a complete DSL string up to 24,000 characters."]}
        outcome = run_dsl_through_pipeline(dsl, require_sdk=True)
        status = "sdk_compiled"
        if not outcome.ok:
            status = {None: "static_invalid", False: "sdk_unavailable", True: "sdk_failed"}[outcome.compiler_available]
        result = {"ok": outcome.ok, "status": status, "normalized_dsl": outcome.dsl_code,
                  "auto_fixes": outcome.auto_fixes, "errors": [str(e) for e in outcome.errors]}
        if outcome.ok:
            issues = _check_mission(outcome.mission_ir, self.task, self.area)
            result.update(ok=not issues, errors=issues)
            if issues:
                result["status"] = "task_mismatch"
            else:
                result["mission_json"] = outcome.mission_ir
                try:
                    result["sdk_version"] = version("steeleagle-sdk")
                except PackageNotFoundError:
                    result["sdk_version"] = "source checkout"
        self.attempts.append(result)
        return result

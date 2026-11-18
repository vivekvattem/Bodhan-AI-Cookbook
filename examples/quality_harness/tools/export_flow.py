import json

from lfx.graph import Graph

from bodhan_eval.components.deterministic import DeterministicChecks
from bodhan_eval.components.judging import Judging
from bodhan_eval.components.model_evidence import ModelEvidence
from bodhan_eval.components.preparation import Preparation
from bodhan_eval.components.validation import ResultValidation
from bodhan_eval.flow import FIRST_NODE, FLOW, LAST_NODE


def main():
    prepare = Preparation()
    checks = DeterministicChecks()
    evidence = ModelEvidence()
    judge = Judging()
    validate = ResultValidation()
    stages = (prepare, checks, evidence, judge, validate)
    ids = (
        FIRST_NODE,
        "DeterministicChecks-harness",
        "ModelEvidence-harness",
        "Judging-harness",
        LAST_NODE,
    )
    graph = Graph()
    for component, component_id in zip(stages, ids, strict=True):
        graph.add_component(component, component_id)
    ports = ("prepared", "checked", "evidence", "reviewed")
    for source, port, target in zip(ids[:-1], ports, ids[1:], strict=True):
        graph.add_component_edge(source, ("batch", port), target)
    graph.initialize()
    export = graph.dump(
        name="Bodhan Quality Harness",
        description="Evaluate supplied or generated outputs.",
    )
    export["id"] = "9159405e-2b03-4d12-b243-58782d0fd987"
    for node in export["data"]["nodes"]:
        node["position"] = {"x": 350 * ids.index(node["id"]), "y": 0}
        node["type"] = "genericNode"
    for edge in export["data"]["edges"]:
        edge["id"] = f"{edge['source']}--{edge['target']}"
        for handle in ("sourceHandle", "targetHandle"):
            edge[handle] = json.dumps(edge["data"][handle]).replace('"', "œ")
    FLOW.write_text(
        json.dumps(export, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

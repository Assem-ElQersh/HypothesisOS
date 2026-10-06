import os
import json
import math
from datetime import datetime
from config import config

class ResearchTree:
    """
    Tree-based persistent memory system for tracking experimental hypotheses,
    branches, outcomes, visits/wins statistics, UCB1 tree search, and synthesis.
    """
    def __init__(self, filepath=config.RESEARCH_TREE_PATH):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "objective": "Minimize validation BPB (bits per byte) on PyTorch language model.",
            "current_best": {
                "val_bpb": float("inf"),
                "val_loss": float("inf"),
                "hypothesis_id": "baseline"
            },
            "root_id": "root",
            "nodes": {
                "root": {
                    "node_id": "root",
                    "parent_id": None,
                    "mechanism": "Initial PyTorch Baseline Model",
                    "val_bpb": None,
                    "val_loss": None,
                    "status": "VALID",
                    "visits": 1,
                    "wins": 1,
                    "children": []
                }
            },
            "active_node_id": "root",
            "proven_wins": [],
            "proven_failures": [],
            "trajectory": []
        }

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def set_baseline(self, val_bpb: float, val_loss: float):
        """ Ground truth baseline metrics setting for root and current_best """
        self.data["current_best"] = {
            "val_bpb": val_bpb,
            "val_loss": val_loss,
            "hypothesis_id": "root"
        }
        if "root" in self.data["nodes"]:
            self.data["nodes"]["root"]["val_bpb"] = val_bpb
            self.data["nodes"]["root"]["val_loss"] = val_loss
            self.data["nodes"]["root"]["visits"] = 1
            self.data["nodes"]["root"]["wins"] = 1
        self.save()

    def add_experiment_node(self, hypothesis: dict, parent_id: str = None) -> str:
        if not parent_id:
            parent_id = self.data.get("active_node_id", "root")

        node_id = hypothesis["id"]
        node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "mechanism": hypothesis["mechanism"],
            "risk": hypothesis.get("risk", "Medium"),
            "patch_diff": hypothesis["code_changes"],
            "utility_ev": hypothesis.get("utility_ev", 0.0),
            "created_at": datetime.now().isoformat(),
            "status": "PENDING",
            "val_bpb": None,
            "val_loss": None,
            "delta_bpb": None,
            "visits": 0,
            "wins": 0,
            "children": [],
            "postmortem": None
        }

        self.data["nodes"][node_id] = node
        if parent_id in self.data["nodes"]:
            if node_id not in self.data["nodes"][parent_id].get("children", []):
                self.data["nodes"][parent_id].setdefault("children", []).append(node_id)
        
        self.save()
        return node_id

    def update_experiment_result(self, node_id: str, results: dict, postmortem: str, baseline_bpb: float):
        if node_id not in self.data["nodes"]:
            return

        node = self.data["nodes"][node_id]
        status = results.get("status", "FAILED")
        val_bpb = results.get("val_bpb")
        val_loss = results.get("val_loss")

        node["status"] = status
        node["val_bpb"] = val_bpb
        node["val_loss"] = val_loss
        node["postmortem"] = postmortem
        node["visits"] = node.get("visits", 0) + 1

        delta_bpb = (val_bpb - baseline_bpb) if (val_bpb is not None and baseline_bpb is not None) else None
        node["delta_bpb"] = round(delta_bpb, 4) if delta_bpb is not None else None

        # Check if experiment is a verified win
        is_win = (status == "VALID" and val_bpb is not None and (baseline_bpb is None or val_bpb < baseline_bpb))

        if is_win:
            node["wins"] = node.get("wins", 0) + 1
            if node_id not in self.data["proven_wins"]:
                self.data["proven_wins"].append(node_id)
            self.data["current_best"] = {
                "val_bpb": val_bpb,
                "val_loss": val_loss,
                "hypothesis_id": node_id
            }
            # Update active node pointer to this winning node
            self.data["active_node_id"] = node_id
        else:
            if node_id not in self.data["proven_failures"]:
                self.data["proven_failures"].append(node_id)

        # Increment parent visit count
        parent_id = node.get("parent_id")
        if parent_id and parent_id in self.data["nodes"]:
            pnode = self.data["nodes"][parent_id]
            pnode["visits"] = pnode.get("visits", 0) + 1
            if is_win:
                pnode["wins"] = pnode.get("wins", 0) + 1

        self.data["trajectory"].append({
            "timestamp": datetime.now().isoformat(),
            "node_id": node_id,
            "status": status,
            "val_bpb": val_bpb,
            "delta_bpb": node["delta_bpb"],
            "is_win": is_win
        })

        self.save()
        self.update_research_plan_file()

    def select_expansion_node(self, c_param: float = 1.414) -> str:
        """ Upper Confidence Bound (UCB1) tree search algorithm for node selection """
        nodes = self.data.get("nodes", {})
        if not nodes:
            return "root"
        
        total_visits = sum(n.get("visits", 1) for n in nodes.values())
        if total_visits == 0:
            total_visits = 1
        
        best_node_id = "root"
        best_ucb = -float("inf")
        
        for nid, node in nodes.items():
            # Only expand valid or root nodes
            if node.get("status") not in ["VALID", "PENDING"] and nid != "root":
                continue
            
            n_i = node.get("visits", 0)
            if n_i == 0:
                # Unvisited valid nodes get top priority
                return nid
            
            w_i = node.get("wins", 0)
            exploitation = w_i / n_i
            exploration = c_param * math.sqrt(math.log(total_visits) / n_i)
            ucb_score = exploitation + exploration
            
            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_node_id = nid
                
        return best_node_id

    def get_research_insights(self) -> dict:
        """ Synthesize structured insights on successful mechanisms and failure patterns """
        wins = []
        failures = []
        for nid, node in self.data.get("nodes", {}).items():
            if nid == "root":
                continue
            mech = node.get("mechanism", "Unknown")
            status = node.get("status", "UNKNOWN")
            delta = node.get("delta_bpb")
            post = node.get("postmortem", "")
            summary = f"[{nid}] Mechanism: {mech} | Status: {status} | Delta BPB: {delta} | Postmortem: {post[:150]}"
            if nid in self.data.get("proven_wins", []):
                wins.append(summary)
            else:
                failures.append(summary)
        return {
            "proven_wins": wins,
            "proven_failures": failures,
            "total_experiments": len(self.data.get("nodes", {})) - 1
        }

    def update_research_plan_file(self):
        best = self.data.get("current_best", {})
        best_bpb = best.get("val_bpb", "N/A")
        best_id = best.get("hypothesis_id", "baseline")
        wins_count = len(self.data.get("proven_wins", []))
        failures_count = len(self.data.get("proven_failures", []))

        content = f"""# Autonomous Research Plan & Trajectory

- **Objective**: Minimize validation BPB on PyTorch language model architecture.
- **Current Best Result**: {best_bpb} BPB (Node: `{best_id}`)
- **Proven Wins Count**: {wins_count}
- **Proven Failures Count**: {failures_count}
- **Active Research Branch**: `{self.data.get("active_node_id", "root")}`

## Research Trajectory History
| Timestamp | Hypothesis ID | Status | Val BPB | Delta BPB | Win? |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for step in reversed(self.data.get("trajectory", [])[-10:]):
            content += f"| {step['timestamp'][:19]} | `{step['node_id']}` | {step['status']} | {step['val_bpb']} | {step['delta_bpb']} | {'YES' if step['is_win'] else 'NO'} |\n"

        with open(config.RESEARCH_PLAN_PATH, "w", encoding="utf-8") as f:
            f.write(content)

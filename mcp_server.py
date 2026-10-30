"""MCP server for Inventor-Argon: lets an AI client query the invention library, novel-invention registry,
simulation results, experience log and neural-net telemetry, and (optionally) drive the running engine.

Read tools work straight from data/*.json (re-read whenever the engine rewrites them), so they need no GPU
and work whether or not run_inventor.py is running. Tools that change state (simulate, review, escalate,
run a wave) go through the live dashboard API so they never race the engine's own writes.

Run:  python mcp_server.py                      (stdio, for Claude Desktop / Claude Code)
      python mcp_server.py --transport http     (streamable HTTP on 127.0.0.1:8765)
Env:  INVENTOR_DATA_DIR (default ./data), INVENTOR_DASHBOARD_URL (default http://127.0.0.1:8050)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("INVENTOR_DATA_DIR", ROOT / "data"))
SIMS_DIR = ROOT / "inventor" / "static" / "sims"
DASHBOARD_URL = os.environ.get("INVENTOR_DASHBOARD_URL", "http://127.0.0.1:8050").rstrip("/")
LIBRARY_PATH = DATA_DIR / "library_state.json"
REGISTRY_PATH = DATA_DIR / "novel_inventions_registry.json"
CHECKPOINT_PATH = DATA_DIR / "inventor_net.pt"

PROPERTY_NAMES = [
    "thermal_stability", "mechanical_strength", "hardness", "electrical_conductivity",
    "thermal_conductivity", "reactivity", "energy_density", "corrosion_resistance",
    "malleability", "optical_transparency", "acidity_basicity", "functional_complexity",
]
SIM_SECTIONS = ["summary", "dft_relaxation", "multiphysics_device", "xrd_peaks", "structure_3d", "raw_keys"]

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

mcp = FastMCP(
    "inventor-argon",
    instructions=(
        "Inventor-Argon is a neural-symbolic invention engine: workers propose recipes (blueprints) built from "
        "library items; a 3-model judge panel votes; accepted recipes become new library items (tiers 0-5). "
        "Items judged NOVEL_FRONTIER go to a novel-inventions registry and an escalation review queue. Items can "
        "be verified by a GPU simulator (CHGNet DFT/MD, cermet micromechanics, device multiphysics).\n"
        "Start with get_overview. Item references accept an item_id (e.g. 'inv_4_cermet_electromagnetic_actua_71') "
        "or an exact/unique-partial name. Property values are normalised 0-1. Use list_items to browse/rank, "
        "get_item for detail, list_simulations/get_simulation for simulation results, and "
        "list_experiences/get_failure_analysis to see what the engine tried and why it failed. "
        "Large simulation arrays are opt-in via get_simulation sections."
    ),
)


# --------------------------------------------------------------------------------------
# Data access (mtime-cached; the engine rewrites these files atomically while running)
# --------------------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def _load_json(path: Path) -> Any:
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        raise ToolError(f"Data file not found: {path}. Has run_inventor.py been run (or is INVENTOR_DATA_DIR set)?")
    key = str(path)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    last_exc: Optional[Exception] = None
    for _ in range(4):  # tolerate a read racing the engine's os.replace
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with _cache_lock:
                _cache[key] = (mtime, data)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            last_exc = exc
            import time
            time.sleep(0.1)
    raise ToolError(f"Could not read {path}: {last_exc}")


class ToolError(Exception):
    """Raised for user-facing tool failures (surfaced to the AI as an error result)."""


def _library() -> Dict[str, Any]:
    return _load_json(LIBRARY_PATH)


def _registry() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return {}
    return {e["item_id"]: e for e in _load_json(REGISTRY_PATH).get("novel_inventions", []) if e.get("item_id")}


def _items() -> Dict[str, Dict[str, Any]]:
    """Library items merged with their novel-registry entry (registry fields under 'registry_*' / flags)."""
    reg = _registry()
    out: Dict[str, Dict[str, Any]] = {}
    for it in _library().get("items", []):
        merged = dict(it)
        r = reg.get(it["item_id"])
        if r:
            merged["in_novel_registry"] = True
            merged["flagged_novel_useful"] = r.get("flagged_novel_useful")
            merged["novel_utility_analysis"] = r.get("novel_utility_analysis")
            merged["discovered_at"] = r.get("discovered_at")
            if not merged.get("simulation_verification") and r.get("simulation_verification"):
                merged["simulation_verification"] = r["simulation_verification"]
            if r.get("blueprint"):
                merged["blueprint"] = r["blueprint"]
        out[it["item_id"]] = merged
    return out


def _resolve(ref: str) -> Dict[str, Any]:
    items = _items()
    if ref in items:
        return items[ref]
    low = ref.strip().lower()
    exact = [i for i in items.values() if i["name"].lower() == low or i.get("formula_or_spec", "").lower() == low]
    if len(exact) == 1:
        return exact[0]
    cands = exact or [i for i in items.values() if low in i["name"].lower() or low in i["item_id"].lower()]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise ToolError(f"No item matches '{ref}'. Use list_items(query=...) to search.")
    names = "; ".join(f"{c['item_id']} ({c['name']})" for c in cands[:10])
    raise ToolError(f"'{ref}' is ambiguous ({len(cands)} matches): {names}")


def _novelty_label(it: Dict[str, Any]) -> str:
    return it.get("novelty_classification") or ("KNOWN_REAL_WORLD" if it.get("is_known_real_world") else "NOVEL_FRONTIER")


def _row(it: Dict[str, Any], props: bool = False) -> Dict[str, Any]:
    sim = it.get("simulation_verification") or {}
    row = {
        "item_id": it["item_id"],
        "name": it["name"],
        "formula": it.get("formula_or_spec"),
        "tier": it.get("tier"),
        "category": it.get("category"),
        "phase": it.get("phase"),
        "capabilities": it.get("capabilities", []),
        "novelty": _novelty_label(it),
        "escalation_status": it.get("escalation_status"),
        "utility_score": it.get("utility_score"),
        "simulated": bool(sim),
        "simulation_passed": sim.get("simulation_passed") if sim else None,
    }
    if props:
        row["properties"] = it.get("properties", {})
    return row


def _clip(text: Optional[str], n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _paginate(rows: List[Any], limit: int, offset: int) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    page = rows[offset: offset + limit]
    return {
        "total_matches": len(rows),
        "offset": offset,
        "returned": len(page),
        "has_more": offset + len(page) < len(rows),
        "results": page,
    }


def _dashboard(path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 600.0) -> Any:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise ToolError(
            f"Inventor-Argon engine dashboard not reachable at {DASHBOARD_URL} ({exc}). "
            "Start it with: python run_inventor.py --serve  (add --continuous for autonomous discovery)."
        )


def _dashboard_up() -> bool:
    try:
        _dashboard("/api/status", timeout=3.0)
        return True
    except ToolError:
        return False


# --------------------------------------------------------------------------------------
# Overview / aggregate tools
# --------------------------------------------------------------------------------------

@mcp.tool(annotations=READ)
def get_overview() -> Dict[str, Any]:
    """High-level snapshot of the engine: library size by tier/category/novelty, hit rate, unlocked equipment gates
    and capabilities, escalation queue, simulation coverage, and whether the live engine is running. Call this first."""
    lib = _library()
    items = _items()
    stats = lib.get("stats", {})
    cats = Counter(i.get("category") for i in items.values())
    nov = Counter(_novelty_label(i) for i in items.values())
    esc = Counter(i.get("escalation_status") for i in items.values())
    sims = [i for i in items.values() if i.get("simulation_verification")]
    return {
        "engine_live": _dashboard_up(),
        "step_counter": lib.get("step_counter"),
        "total_items": len(items),
        "by_tier": dict(sorted(Counter(i.get("tier") for i in items.values()).items())),
        "by_category": dict(cats.most_common()),
        "by_novelty": dict(nov),
        "by_escalation_status": {k: v for k, v in esc.items() if k},
        "evaluations": {
            "total": stats.get("total_evaluations"),
            "valid": stats.get("valid_blueprints"),
            "hit_rate": stats.get("hit_rate"),
            "new_discoveries": stats.get("new_discoveries"),
        },
        "capabilities_unlocked": stats.get("capabilities_unlocked"),
        "equipment_gates": {k: {"unlocked": v.get("unlocked"), "by": v.get("equipment")} for k, v in stats.get("equipment_gates", {}).items()},
        "novel_registry_size": len(_registry()),
        "simulations": {
            "simulated_items": len(sims),
            "passed": sum(1 for i in sims if i["simulation_verification"].get("simulation_passed")),
            "item_ids": [i["item_id"] for i in sims],
        },
        "data_files": {"library": str(LIBRARY_PATH), "registry": str(REGISTRY_PATH), "checkpoint": str(CHECKPOINT_PATH)},
    }


@mcp.tool(annotations=READ)
def get_equipment_gates() -> Dict[str, Any]:
    """Equipment/tech gates (wet chemistry, high-temp furnace, electrochemistry, high-pressure, precision fabrication,
    plasma arc): which are unlocked and by which invented item. Gates constrain what recipes can use."""
    return _library().get("stats", {}).get("equipment_gates", {})


@mcp.tool(annotations=READ)
def get_capability_coverage() -> Dict[str, Any]:
    """For each functional capability (structural, refractory, conductor, semiconductor, ...): how many items provide it
    and the top few by tier. Useful for spotting gaps in what the library can do."""
    by_cap: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in _items().values():
        for c in it.get("capabilities", []):
            by_cap[c].append(it)
    return {
        cap: {
            "count": len(lst),
            "highest_tier": max(i.get("tier", 0) for i in lst),
            "top_items": [f"{i['name']} (tier {i.get('tier')}, {i['item_id']})" for i in sorted(lst, key=lambda x: -x.get("tier", 0))[:5]],
        }
        for cap, lst in sorted(by_cap.items())
    }


@mcp.tool(annotations=READ)
def get_concept_families(min_count: int = 1) -> Dict[str, int]:
    """Concept-family saturation counts (how many items share a family such as 'oxide_compound'). High counts indicate
    areas the engine has over-explored."""
    fam = _library().get("stats", {}).get("concept_families", {})
    return dict(sorted(((k, v) for k, v in fam.items() if v >= min_count), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------

@mcp.tool(annotations=READ)
def list_items(
    query: Optional[str] = None,
    tier: Optional[int] = None,
    min_tier: Optional[int] = None,
    category: Optional[str] = None,
    capability: Optional[str] = None,
    novelty: Optional[Literal["KNOWN_REAL_WORLD", "NOVEL_FRONTIER"]] = None,
    escalation_status: Optional[str] = None,
    simulated: Optional[bool] = None,
    min_properties: Optional[Dict[str, float]] = None,
    sort_by: str = "tier",
    descending: bool = True,
    include_properties: bool = False,
    limit: int = 25,
    offset: int = 0,
) -> Dict[str, Any]:
    """Browse, search and rank library items.

    query: case-insensitive substring over name, formula, description, id.
    tier / min_tier: 0=elements/primitives ... 5=most complex devices.
    capability: e.g. structural, refractory, conductor, semiconductor, magnetic, actuator.
    min_properties: e.g. {"hardness": 0.8, "thermal_stability": 0.9} (values are 0-1).
    sort_by: 'tier', 'name', 'utility_score', 'discovery_step', or any property name (hardness, ...).
    """
    q = query.lower() if query else None
    rows = []
    for it in _items().values():
        if q and q not in f"{it['name']} {it.get('formula_or_spec', '')} {it.get('description', '')} {it['item_id']}".lower():
            continue
        if tier is not None and it.get("tier") != tier:
            continue
        if min_tier is not None and it.get("tier", 0) < min_tier:
            continue
        if category and (it.get("category") or "").lower() != category.lower():
            continue
        if capability and capability.lower() not in [c.lower() for c in it.get("capabilities", [])]:
            continue
        if novelty and _novelty_label(it) != novelty:
            continue
        if escalation_status and it.get("escalation_status") != escalation_status:
            continue
        if simulated is not None and bool(it.get("simulation_verification")) != simulated:
            continue
        if min_properties and any(it.get("properties", {}).get(k, 0.0) < v for k, v in min_properties.items()):
            continue
        rows.append(it)

    def key(it: Dict[str, Any]) -> Any:
        if sort_by in PROPERTY_NAMES:
            return it.get("properties", {}).get(sort_by, 0.0)
        if sort_by == "name":
            return it["name"].lower()
        return it.get(sort_by) or 0

    if sort_by not in PROPERTY_NAMES + ["tier", "name", "utility_score", "discovery_step"]:
        raise ToolError(f"Unknown sort_by '{sort_by}'. Use tier, name, utility_score, discovery_step or one of {PROPERTY_NAMES}")
    rows.sort(key=key, reverse=descending)
    out = _paginate([_row(r, include_properties) for r in rows], limit, offset)
    return out


@mcp.tool(annotations=READ)
def get_item(item_id: str, include_judge_ballots: bool = False) -> Dict[str, Any]:
    """Full detail for one item (id or name): properties, capabilities, recipe, parents, novelty rationale, deep audit,
    practical applications, and a simulation summary (use get_simulation for the full simulation data)."""
    it = _resolve(item_id)
    sim = it.get("simulation_verification") or {}
    jv = it.get("judge_votes") or {}
    out = {
        **_row(it, props=True),
        "description": it.get("description"),
        "elemental_lineage": it.get("elemental_lineage"),
        "parent_ids": it.get("parent_ids", []),
        "recipe_summary": it.get("recipe_summary"),
        "created_by_blueprint_id": it.get("created_by_blueprint_id"),
        "discovered_by_worker": it.get("discovered_by_worker"),
        "discovery_step": it.get("discovery_step"),
        "times_used_in_valid_recipes": it.get("times_used_in_valid_recipes"),
        "novelty_rationale": it.get("novelty_rationale"),
        "real_world_equivalent": it.get("real_world_equivalent"),
        "real_world_interpretation": it.get("real_world_interpretation"),
        "practical_applications": it.get("practical_applications"),
        "deep_audit": it.get("deep_audit"),
        "judge_decision": {k: jv.get(k) for k in ("decision", "votes_yes", "total_judges", "novel_votes", "track")} if jv else None,
        "novel_utility_analysis": it.get("novel_utility_analysis"),
        "simulation": (
            {"mode": sim.get("simulation_mode"), "passed": sim.get("simulation_passed"), "verdict": sim.get("summary_verdict"),
             "detail_tool": f"get_simulation('{it['item_id']}')"} if sim else None
        ),
    }
    if include_judge_ballots and jv:
        out["judge_ballots"] = jv.get("ballots")
    return out


@mcp.tool(annotations=READ)
def get_lineage(item_id: str, direction: Literal["ancestors", "descendants", "both"] = "both", max_depth: int = 4) -> Dict[str, Any]:
    """Trace how an item was made (ancestors via parent_ids back to elements) and/or what was made from it
    (descendants). Each node shows id, name, tier."""
    items = _items()
    root = _resolve(item_id)
    children: Dict[str, List[str]] = defaultdict(list)
    for i in items.values():
        for p in i.get("parent_ids", []):
            children[p].append(i["item_id"])

    def node(i: Dict[str, Any]) -> Dict[str, Any]:
        return {"item_id": i["item_id"], "name": i["name"], "formula": i.get("formula_or_spec"), "tier": i.get("tier")}

    def up(i: Dict[str, Any], d: int, seen: frozenset) -> Dict[str, Any]:
        n = node(i)
        if d < max_depth and i.get("parent_ids"):
            n["made_from"] = [up(items[p], d + 1, seen | {i["item_id"]}) for p in i["parent_ids"] if p in items and p not in seen]
        return n

    def down(i: Dict[str, Any], d: int, seen: frozenset) -> Dict[str, Any]:
        n = node(i)
        kids = [c for c in children.get(i["item_id"], []) if c not in seen]
        if kids:
            if d < max_depth:
                n["used_to_make"] = [down(items[c], d + 1, seen | {i["item_id"]}) for c in kids[:15]]
                if len(kids) > 15:
                    n["used_to_make_truncated"] = len(kids) - 15
            else:
                n["used_to_make_count"] = len(kids)
        return n

    out: Dict[str, Any] = {"item": node(root), "recipe_summary": root.get("recipe_summary")}
    if direction in ("ancestors", "both"):
        out["ancestry"] = up(root, 0, frozenset())
    if direction in ("descendants", "both"):
        out["descendants"] = down(root, 0, frozenset())
    return out


@mcp.tool(annotations=READ)
def compare_items(item_ids: List[str]) -> Dict[str, Any]:
    """Side-by-side comparison of 2-8 items: all 12 properties, tier, capabilities, utility and simulation status,
    plus which item is best for each property."""
    if not 2 <= len(item_ids) <= 8:
        raise ToolError("Provide between 2 and 8 items.")
    its = [_resolve(r) for r in item_ids]
    table = {p: {i["name"]: i.get("properties", {}).get(p) for i in its} for p in PROPERTY_NAMES}
    best = {p: max(vals, key=lambda n: vals[n] if vals[n] is not None else -1) for p, vals in table.items()}
    return {
        "items": [_row(i) for i in its],
        "properties": table,
        "best_per_property": best,
        "shared_capabilities": sorted(set.intersection(*[set(i.get("capabilities", [])) for i in its])),
    }


# --------------------------------------------------------------------------------------
# Novel inventions & escalation
# --------------------------------------------------------------------------------------

@mcp.tool(annotations=READ)
def list_novel_inventions(
    flagged_useful_only: bool = False,
    escalation_status: Optional[str] = None,
    sort_by: Literal["newest", "utility_score", "tier"] = "newest",
    limit: int = 25,
    offset: int = 0,
) -> Dict[str, Any]:
    """The novel-frontier registry: inventions judged not to match known real-world materials. Filter to those flagged
    novel-and-useful or by escalation status (PENDING_ESCALATION, VERIFIED_NOVEL_BREAKTHROUGH,
    PLAUSIBLE_FRONTIER_CONSTRUCT, RECLASSIFIED_KNOWN, ...)."""
    rows = list(_registry().values())
    if flagged_useful_only:
        rows = [r for r in rows if r.get("flagged_novel_useful")]
    if escalation_status:
        rows = [r for r in rows if r.get("escalation_status") == escalation_status]
    keyf = {"newest": lambda r: r.get("discovered_at", 0), "utility_score": lambda r: r.get("utility_score") or 0, "tier": lambda r: r.get("tier", 0)}[sort_by]
    rows.sort(key=keyf, reverse=True)
    slim = [{
        "item_id": r["item_id"], "name": r["name"], "formula": r.get("formula_or_spec"), "tier": r.get("tier"),
        "category": r.get("category"), "escalation_status": r.get("escalation_status"),
        "flagged_novel_useful": r.get("flagged_novel_useful"), "utility_score": r.get("utility_score"),
        "real_world_equivalent": r.get("real_world_equivalent"), "judge_decision": (r.get("judge_votes") or {}).get("decision"),
        "simulated": bool(r.get("simulation_verification")), "strategy": r.get("strategy"),
    } for r in rows]
    return _paginate(slim, limit, offset)


@mcp.tool(annotations=READ)
def get_novel_invention(item_id: str) -> Dict[str, Any]:
    """Everything the registry holds for a novel invention: recipe, per-judge ballots and novelty rationales,
    Chief-Scientist deep audit (thermodynamics/math, novelty vs known world), utility analysis and applications."""
    it = _resolve(item_id)
    r = _registry().get(it["item_id"])
    if not r:
        raise ToolError(f"{it['item_id']} is not in the novel registry (novelty: {_novelty_label(it)}).")
    out = {k: v for k, v in r.items() if k not in ("simulation_verification", "blueprint")}
    out["blueprint_summary"] = _blueprint_summary(r.get("blueprint") or {})
    out["simulation"] = (r.get("simulation_verification") or {}).get("summary_verdict")
    return out


@mcp.tool(annotations=READ)
def get_escalation_queue() -> Dict[str, Any]:
    """Items awaiting expert review (escalation_status == PENDING_ESCALATION) plus counts of every review status."""
    items = _items().values()
    pending = [_row(i) | {"novelty_rationale": _clip(i.get("novelty_rationale"), 400)} for i in items if i.get("escalation_status") == "PENDING_ESCALATION"]
    return {"status_counts": dict(Counter(i.get("escalation_status") for i in items)), "pending": pending}


# --------------------------------------------------------------------------------------
# Simulation results
# --------------------------------------------------------------------------------------

def _sim_of(it: Dict[str, Any]) -> Dict[str, Any]:
    sim = it.get("simulation_verification")
    if not sim:
        raise ToolError(f"{it['item_id']} has not been simulated. Use run_simulation('{it['item_id']}') (needs the engine/GPU).")
    return sim


@mcp.tool(annotations=READ)
def list_simulations(passed: Optional[bool] = None, mode: Optional[str] = None) -> Dict[str, Any]:
    """All items that have simulation results, with mode (CRYSTAL_DFT_MLIP_AND_MD, CERMET_PHASE_AND_MICROMECHANICS,
    DEVICE_MULTIPHYSICS_AND_MATERIAL), pass/fail, runtime and the one-line verdict."""
    out = []
    for it in _items().values():
        sim = it.get("simulation_verification")
        if not sim:
            continue
        if passed is not None and bool(sim.get("simulation_passed")) != passed:
            continue
        if mode and sim.get("simulation_mode") != mode:
            continue
        out.append({
            "item_id": it["item_id"], "name": it["name"], "formula": it.get("formula_or_spec"), "tier": it.get("tier"),
            "mode": sim.get("simulation_mode"), "engine": sim.get("engine"), "passed": sim.get("simulation_passed"),
            "runtime_s": sim.get("runtime_s"), "verdict": sim.get("summary_verdict"),
            "has_cif": bool(sim.get("cif_url")),
        })
    return {"count": len(out), "simulations": out}


@mcp.tool(annotations=READ)
def get_simulation(
    item_id: str,
    sections: Optional[List[Literal["summary", "dft_relaxation", "multiphysics_device", "xrd_peaks", "structure_3d", "raw_keys"]]] = None,
    max_trajectory_points: int = 20,
) -> Dict[str, Any]:
    """Simulation results for an item. `sections` selects what to return (default: summary, dft_relaxation,
    multiphysics_device, xrd_peaks). 'structure_3d' (atom coords/bonds) is large and opt-in; 'raw_keys' lists
    every top-level key present (modes differ: crystal runs have MD/hull data, cermets have micromechanics, devices
    have trajectories). Long time-series are down-sampled to max_trajectory_points."""
    it = _resolve(item_id)
    sim = _sim_of(it)
    sections = sections or ["summary", "dft_relaxation", "multiphysics_device", "xrd_peaks"]
    out: Dict[str, Any] = {"item_id": it["item_id"], "name": it["name"], "mode": sim.get("simulation_mode")}
    if "summary" in sections:
        out["summary"] = {k: sim.get(k) for k in ("engine", "simulation_passed", "runtime_s", "summary_verdict", "cif_url")}
    if "raw_keys" in sections:
        out["raw_keys"] = list(sim.keys())
    handled = {"simulation_mode", "engine", "simulation_passed", "runtime_s", "summary_verdict", "cif_url", "xrd_peaks", "structure_3d"}
    if "dft_relaxation" in sections:
        # every mode-specific block except the big ones (dft_relaxation, md, hull, micromechanics, ...)
        out["dft_relaxation"] = sim.get("dft_relaxation")
        handled.add("dft_relaxation")
    if "multiphysics_device" in sections:
        mp = sim.get("multiphysics_device")
        out["multiphysics_device"] = _downsample_lists(mp, max_trajectory_points) if mp else None
        handled.add("multiphysics_device")
    # surface any other mode-specific blocks (MD stats, hull, micromechanics...) alongside the requested sections
    if "dft_relaxation" in sections or "summary" in sections:
        extra = {k: _downsample_lists(v, max_trajectory_points) for k, v in sim.items() if k not in handled}
        if extra:
            out["other_results"] = extra
    if "xrd_peaks" in sections:
        out["xrd_peaks"] = sim.get("xrd_peaks")
    if "structure_3d" in sections:
        s3 = sim.get("structure_3d") or {}
        out["structure_3d"] = {"n_atoms": len(s3.get("atoms", [])), "atoms": s3.get("atoms"), "bonds": s3.get("bonds")}
    return out


def _downsample_lists(obj: Any, n: int) -> Any:
    """Recursively thin long lists to n evenly spaced points (keeping first and last)."""
    if isinstance(obj, dict):
        return {k: _downsample_lists(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > n > 1:
            idx = sorted({round(i * (len(obj) - 1) / (n - 1)) for i in range(n)})
            return {"_downsampled_from": len(obj), "points": [_downsample_lists(obj[i], n) for i in idx]}
        return [_downsample_lists(v, n) for v in obj]
    return obj


@mcp.tool(annotations=READ)
def get_simulation_timeseries(item_id: str, series: Optional[str] = None, max_points: int = 200) -> Dict[str, Any]:
    """Full-resolution time-series/curve data from a simulation (e.g. solenoid 'dynamics_trajectory' of stroke/current/force
    vs time, 'archard_wear_comparison'). Omit `series` to list available series names and lengths."""
    sim = _sim_of(_resolve(item_id))
    found: Dict[str, List[Any]] = {}

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list) and o and isinstance(o[0], dict) and path not in ("structure_3d.atoms",):
            found[path] = o

    walk({k: v for k, v in sim.items() if k != "structure_3d"}, "")
    if not series:
        return {"available_series": {k: {"points": len(v), "fields": list(v[0].keys())} for k, v in found.items()}}
    matches = [k for k in found if k == series or k.endswith("." + series) or k.endswith(series)]
    if not matches:
        raise ToolError(f"No series '{series}'. Available: {list(found)}")
    data = found[matches[0]]
    step = max(1, -(-len(data) // max_points))
    return {"series": matches[0], "total_points": len(data), "returned": len(data[::step]), "data": data[::step]}


@mcp.tool(annotations=READ)
def get_cif(item_id: str) -> Dict[str, Any]:
    """The relaxed crystal structure for a simulated item as CIF text (usable in VESTA/pymatgen/ASE), if one was written."""
    it = _resolve(item_id)
    path = SIMS_DIR / f"{it['item_id']}.cif"
    if not path.is_file():
        raise ToolError(f"No CIF for {it['item_id']}. Simulated items with CIFs: {sorted(p.stem for p in SIMS_DIR.glob('*.cif'))}")
    return {"item_id": it["item_id"], "path": str(path), "cif": path.read_text(encoding="utf-8")}


@mcp.tool(annotations=READ)
def simulation_property_summary(mode: Optional[str] = None) -> Dict[str, Any]:
    """Numeric scalar results flattened across all simulated items (energies, moments, forces, bulk moduli, ...), one
    table per simulation mode, for cross-item comparison."""
    tables: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for it in _items().values():
        sim = it.get("simulation_verification")
        if not sim or (mode and sim.get("simulation_mode") != mode):
            continue
        flat: Dict[str, Any] = {}

        def walk(o: Any, path: str) -> None:
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("structure_3d", "xrd_peaks"):
                        continue
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(o, (int, float)):
                flat[path] = o

        walk(sim, "")
        tables[sim.get("simulation_mode", "?")][it["name"]] = flat
    return dict(tables)


# --------------------------------------------------------------------------------------
# Experiences / blueprints / judging
# --------------------------------------------------------------------------------------

def _blueprint_summary(bp: Dict[str, Any]) -> Dict[str, Any]:
    if not bp:
        return {}
    return {
        "blueprint_id": bp.get("blueprint_id"), "proposer": bp.get("proposer_id"), "strategy": bp.get("strategy"),
        "goal": bp.get("target_goal"),
        "ingredients": [f"{i.get('ratio')}x {i.get('item_name')} ({i.get('formula')}) as {i.get('role')}" for i in bp.get("ingredients", [])],
        "steps": [
            f"{s.get('step_order')}. {s.get('operation')} @ {s.get('temperature_c')}C / {s.get('pressure_atm')} atm / {s.get('atmosphere')} / {s.get('duration')}"
            + (f" [field: {s['excitation_field']}]" if s.get("excitation_field") not in (None, "none") else "")
            for s in bp.get("steps", [])
        ],
        "equipment": bp.get("equipment_used"),
    }


def _exp_row(e: Dict[str, Any]) -> Dict[str, Any]:
    bp, v = e.get("blueprint", {}), e.get("verdict", {})
    return {
        "blueprint_id": bp.get("blueprint_id"), "worker": bp.get("proposer_id"), "strategy": bp.get("strategy"),
        "goal": bp.get("target_goal"), "valid": v.get("valid"), "is_new_discovery": e.get("is_new_discovery"),
        "discovered_item_id": e.get("discovered_item_id"), "output": v.get("output_name"), "formula": v.get("formula_or_spec"),
        "physics_score": v.get("physics_score"), "reward": e.get("reward"), "judge_decision": (v.get("judge_votes") or {}).get("decision"),
        "failure_reason": _clip(v.get("failure_reason"), 240) or None,
        "predicted_validity": (bp.get("neural_prediction") or {}).get("predicted_validity"),
        "timestamp": e.get("timestamp"),
    }


@mcp.tool(annotations=READ)
def list_experiences(
    valid: Optional[bool] = None,
    new_discovery: Optional[bool] = None,
    strategy: Optional[str] = None,
    worker: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
) -> Dict[str, Any]:
    """Recent evaluated recipes (the engine keeps the latest 250), newest first: who proposed it, the judge decision,
    reward, the neural net's predicted validity vs. the actual outcome, and the failure reason if rejected."""
    exps = list(reversed(_library().get("recent_experiences", [])))
    rows = []
    for e in exps:
        r = _exp_row(e)
        if valid is not None and bool(r["valid"]) != valid:
            continue
        if new_discovery is not None and bool(r["is_new_discovery"]) != new_discovery:
            continue
        if strategy and r["strategy"] != strategy:
            continue
        if worker and worker not in (r["worker"] or ""):
            continue
        if query and query.lower() not in json.dumps(e).lower():
            continue
        rows.append(r)
    return _paginate(rows, limit, offset)


@mcp.tool(annotations=READ)
def get_experience(blueprint_id: str, include_ballots: bool = True) -> Dict[str, Any]:
    """Full record for one evaluated recipe: ingredients/steps/equipment, neural-net prediction (policy outputs and
    predicted properties), each judge's ballot, repair hint, and the resulting item's properties."""
    for e in _library().get("recent_experiences", []):
        if e.get("blueprint", {}).get("blueprint_id") == blueprint_id:
            v = dict(e.get("verdict", {}))
            jv = v.pop("judge_votes", None) or {}
            out = {
                "recipe": _blueprint_summary(e["blueprint"]),
                "neural_prediction": e["blueprint"].get("neural_prediction"),
                "verdict": {k: v.get(k) for k in v if k not in ("simulation_verification",)},
                "judge_decision": {k: jv.get(k) for k in ("decision", "votes_yes", "total_judges", "novel_votes", "track")},
                "reward": e.get("reward"), "is_new_discovery": e.get("is_new_discovery"),
                "discovered_item_id": e.get("discovered_item_id"),
            }
            if include_ballots:
                out["ballots"] = jv.get("ballots")
            return out
    raise ToolError(f"Blueprint '{blueprint_id}' not in the retained experience window (latest 250).")


@mcp.tool(annotations=READ)
def get_failure_analysis(top_n: int = 15) -> Dict[str, Any]:
    """Why recipes fail: hit rate by strategy and worker, most common rejection reasons (keyword-clustered), most
    common failure op/temperature/atmosphere combos, and how well the neural net's predicted validity separates
    valid from invalid recipes. Over the retained experience window."""
    exps = _library().get("recent_experiences", [])
    by_strat: Dict[str, List[bool]] = defaultdict(list)
    by_worker: Dict[str, List[bool]] = defaultdict(list)
    reasons: Counter = Counter()
    combos: Counter = Counter()
    pv_valid, pv_invalid = [], []
    keywords = ["melt", "decompos", "oxid", "volatil", "stoichiometr", "thermodynamic", "unreact", "equipment", "atmosphere",
                "pressure", "temperature", "slag", "mixture", "hydrolys", "impur", "contaminat", "exotherm", "vaporiz"]
    for e in exps:
        bp, v = e.get("blueprint", {}), e.get("verdict", {})
        ok = bool(v.get("valid"))
        by_strat[bp.get("strategy", "?")].append(ok)
        by_worker[bp.get("proposer_id", "?")].append(ok)
        pv = (bp.get("neural_prediction") or {}).get("predicted_validity")
        if pv is not None:
            (pv_valid if ok else pv_invalid).append(pv)
        if not ok:
            text = (v.get("failure_reason") or "").lower()
            for k in keywords:
                if k in text:
                    reasons[k] += 1
            for s in bp.get("steps", []):
                combos[f"{s.get('operation')} @ {s.get('temperature_c')}C in {s.get('atmosphere')}"] += 1

    def rate(d: Dict[str, List[bool]]) -> Dict[str, Any]:
        return {k: {"n": len(x), "hit_rate": round(sum(x) / len(x), 3)} for k, x in sorted(d.items())}

    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {
        "window": len(exps),
        "hit_rate_by_strategy": rate(by_strat),
        "hit_rate_by_worker": rate(by_worker),
        "failure_keyword_counts": dict(reasons.most_common(top_n)),
        "most_common_failed_step_combos": dict(combos.most_common(top_n)),
        "neural_net_calibration": {"mean_predicted_validity_when_valid": avg(pv_valid), "mean_predicted_validity_when_invalid": avg(pv_invalid)},
    }


# --------------------------------------------------------------------------------------
# Neural net / judges
# --------------------------------------------------------------------------------------

@mcp.tool(annotations=READ)
def get_brain_telemetry(include_loss_history: bool = False) -> Dict[str, Any]:
    """Neural-net training state: device, train steps, replay size, latest losses/validity accuracy/property MAE, top
    compatible ingredient pairs. Uses the live engine when running; otherwise reads the checkpoint on disk."""
    if _dashboard_up():
        st = _dashboard("/api/status", timeout=15)
        t = dict(st.get("brain_telemetry", {}))
        t["source"] = "live_engine"
        t["judge_stats"] = st.get("judge_stats")
        t["autonomous_waves_completed"] = st.get("autonomous_waves_completed")
        t["continuous_active"] = st.get("continuous_active")
        if not include_loss_history:
            t.pop("loss_history", None)
        return t
    if not CHECKPOINT_PATH.exists():
        raise ToolError(f"No live engine and no checkpoint at {CHECKPOINT_PATH}.")
    try:
        import torch  # lazy: only needed for the offline path
        ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ToolError(f"Could not read checkpoint: {exc}")
    hist = ck.get("loss_history", [])
    out = {
        "source": "checkpoint_on_disk", "train_steps": ck.get("train_steps"), "latest_metrics": hist[-1] if hist else {},
        "checkpoint_mtime": CHECKPOINT_PATH.stat().st_mtime,
        "n_parameters": sum(int(t.numel()) for t in ck.get("model_state", {}).values()),
    }
    if include_loss_history:
        out["loss_history"] = hist
    return out


@mcp.tool(annotations=READ)
def get_learning_curve(metric: str = "loss_total", points: int = 30) -> Dict[str, Any]:
    """A metric over training history (loss_total, validity_acc, property_mae, ...), down-sampled, with first/last/min/max
    so you can tell whether the net is improving."""
    hist = get_brain_telemetry(include_loss_history=True).get("loss_history") or []
    vals = [(h.get("train_steps"), h[metric]) for h in hist if metric in h]
    if not vals:
        avail = sorted({k for h in hist for k in h})
        raise ToolError(f"Metric '{metric}' not found. Available: {avail}")
    step = max(1, len(vals) // points)
    nums = [v for _, v in vals]
    return {"metric": metric, "n": len(vals), "first": nums[0], "last": nums[-1], "min": min(nums), "max": max(nums),
            "series": [{"train_steps": s, metric: v} for s, v in vals[::step]]}


@mcp.tool(annotations=READ)
def get_judge_stats() -> Dict[str, Any]:
    """3-model judge panel statistics (unanimous/majority/rejected counts, latency, cache hits). Requires the live engine;
    offline, falls back to vote tallies computed from retained experiences."""
    if _dashboard_up():
        return _dashboard("/api/status", timeout=15).get("judge_stats", {})
    tally: Counter = Counter()
    per_model: Dict[str, Counter] = defaultdict(Counter)
    for e in _library().get("recent_experiences", []):
        jv = (e.get("verdict") or {}).get("judge_votes") or {}
        tally[jv.get("decision", "N/A")] += 1
        for name, b in (jv.get("ballots") or {}).items():
            per_model[name][b.get("vote", "?")] += 1
    return {"source": "computed_offline", "decisions": dict(tally), "votes_by_model": {k: dict(v) for k, v in per_model.items()}}


# --------------------------------------------------------------------------------------
# Live engine control (all go through the dashboard API)
# --------------------------------------------------------------------------------------

@mcp.tool(annotations=READ)
def get_engine_status() -> Dict[str, Any]:
    """Is the engine (run_inventor.py --serve) reachable, is the continuous discovery loop active, and how many waves
    has it run? Control tools below require it to be reachable."""
    if not _dashboard_up():
        return {"engine_live": False, "dashboard_url": DASHBOARD_URL, "hint": "python run_inventor.py --serve --continuous"}
    st = _dashboard("/api/status", timeout=15)
    return {"engine_live": True, "dashboard_url": DASHBOARD_URL, "continuous_active": st.get("continuous_active"),
            "autonomous_waves_completed": st.get("autonomous_waves_completed"), "library_stats": {
                k: st["library_stats"].get(k) for k in ("total_library_items", "invented_items", "novel_frontier_items", "pending_escalation_reviews", "hit_rate")}}


@mcp.tool(annotations=WRITE)
def run_simulation(item_id: str) -> Dict[str, Any]:
    """Run the GPU atomistic/multiphysics simulation for an item and store the result on it (takes seconds to minutes;
    overwrites any previous result). Needs the live engine. Returns the summary; fetch details with get_simulation."""
    it = _resolve(item_id)
    res = _dashboard("/api/simulate", {"item_id": it["item_id"]})
    sim = res.get("simulation") or {}
    if not res.get("ok") or not sim:
        raise ToolError(f"Simulation failed or returned nothing for {it['item_id']}.")
    return {"item_id": it["item_id"], "mode": sim.get("simulation_mode"), "passed": sim.get("simulation_passed"),
            "runtime_s": sim.get("runtime_s"), "verdict": sim.get("summary_verdict"), "cif_url": sim.get("cif_url"),
            "next": f"get_simulation('{it['item_id']}')"}


@mcp.tool(annotations=WRITE)
def escalate_item(item_id: str) -> Dict[str, Any]:
    """Put an item in the PENDING_ESCALATION queue for deep first-principles review."""
    it = _resolve(item_id)
    res = _dashboard("/api/escalate", {"item_id": it["item_id"]})
    return {"item_id": it["item_id"], "ok": bool(res.get("ok")), "new_status": "PENDING_ESCALATION" if res.get("ok") else None}


@mcp.tool(annotations=WRITE)
def submit_review(
    item_id: str,
    escalation_status: Literal["VERIFIED_NOVEL_BREAKTHROUGH", "PLAUSIBLE_FRONTIER_CONSTRUCT", "RECLASSIFIED_KNOWN", "PENDING_ESCALATION"],
    verdict: str,
    reviewed_by: str = "MCP client (AI reviewer)",
) -> Dict[str, Any]:
    """Record a review decision on an item. RECLASSIFIED_KNOWN marks it as a known real-world material; the other
    statuses keep it NOVEL_FRONTIER. `verdict` is the free-text rationale stored in the item's deep audit. This
    changes library state: only call it when the user asks you to rule on an item."""
    it = _resolve(item_id)
    res = _dashboard("/api/review", {"item_id": it["item_id"], "escalation_status": escalation_status,
                                     "chief_scientist_verdict": verdict, "reviewed_by": reviewed_by})
    return {"item_id": it["item_id"], "ok": bool(res.get("ok")), "escalation_status": escalation_status}


@mcp.tool(annotations=WRITE)
def run_discovery_wave() -> Dict[str, Any]:
    """Kick off one discovery wave (all workers propose and judge recipes in parallel) in the background. This calls
    external judge LLMs and spends API credits. Poll get_overview / list_experiences for the results."""
    res = _dashboard("/api/run_wave", {})
    return {"started": bool(res.get("ok")), "note": "Runs in the background; check list_experiences(new_discovery=True) in ~30-90s."}


@mcp.tool(annotations=WRITE)
def toggle_continuous_discovery() -> Dict[str, Any]:
    """Toggle the engine's autonomous continuous-discovery loop on/off (only effective if it was started with --serve).
    Returns the new state."""
    _dashboard("/api/toggle_continuous", {})
    return {"continuous_active": _dashboard("/api/status", timeout=15).get("continuous_active")}


# --------------------------------------------------------------------------------------
# Resources (browsable context for clients that support them)
# --------------------------------------------------------------------------------------

@mcp.resource("inventor://overview", mime_type="application/json")
def overview_resource() -> str:
    return json.dumps(get_overview(), indent=2)


@mcp.resource("inventor://item/{item_id}", mime_type="application/json")
def item_resource(item_id: str) -> str:
    return json.dumps(get_item(item_id, include_judge_ballots=True), indent=2)


@mcp.resource("inventor://cif/{item_id}", mime_type="chemical/x-cif")
def cif_resource(item_id: str) -> str:
    return get_cif(item_id)["cif"]


@mcp.prompt()
def analyze_breakthroughs() -> str:
    """Review the engine's most promising novel inventions."""
    return (
        "Using the inventor-argon tools: call get_overview, then list_novel_inventions(flagged_useful_only=True). "
        "For the top candidates call get_novel_invention and get_simulation. Assess scientific plausibility "
        "(thermodynamics, phase stability, whether the simulation actually supports the claim), flag anything that is "
        "really a known material, and rank them. Do not call submit_review unless I ask."
    )


@mcp.prompt()
def diagnose_engine() -> str:
    """Diagnose why the engine is or isn't discovering things."""
    return (
        "Using the inventor-argon tools: call get_overview, get_failure_analysis, get_learning_curve and "
        "get_concept_families. Explain what limits discovery (strategy hit rates, over-saturated families, equipment "
        "gates, net calibration) and suggest concrete changes."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventor-Argon MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.transport == "http":
        mcp.settings.host, mcp.settings.port = args.host, args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio: never print to stdout


if __name__ == "__main__":
    main()

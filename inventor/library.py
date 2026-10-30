"""Thread-safe Shared Knowledge Library, Strict Viability Filter, and Tool-Gated Progression."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from inventor.primitives import get_starting_primitives
from inventor.schemas import (
    CATEGORY_TO_TIER,
    Blueprint,
    ExperienceRecord,
    JudgeVerdict,
    LibraryItem,
    derive_real_world_utility,
    extract_concept_family,
    normalize_item_name,
)

BANNED_NAME_SUBSTRINGS = (
    "/",
    "+",
    " with ",
    "scale",
    "slag",
    "mixture",
    "residual",
    "excess",
    "oxidized ",
    "impure",
    "unreacted",
    "inclusion",
    "multi-phase",
    "hybrid",
    "approximate",
    "mixed",
    "derived",
    "system",
    "clinker",
)


def is_viable_construct(
    name: str,
    formula: str,
    tier: int,
    category: str,
    description: str,
    bp: Optional[Blueprint] = None,
) -> Tuple[bool, str]:
    """Programmatic hard gate rejecting slag, surface scale, multi-phase soups, and flawed conditions."""
    if not name or len(name.strip()) < 2:
        return False, "Empty output name"

    low_name = name.lower().strip()
    low_form = formula.lower().strip()
    low_desc = description.lower().strip()

    # 1. Reject banned junk/mixture/slag substrings in name
    for bad in BANNED_NAME_SUBSTRINGS:
        if bad in low_name:
            return False, f"Rejected non-viable construct name containing '{bad}': {name}"

    if "slag" in low_desc or "surface scale" in low_desc or "unreacted" in low_desc:
        return False, f"Rejected construct containing slag/scale/unreacted phases: {name}"

    # 2. Reject 3+ hyphenated multi-element soups (e.g. Lithium-Copper-Iron-Aluminate)
    clean_words = re.sub(r"\([^)]*\)", "", name).strip()
    if clean_words.count("-") >= 3:
        return False, f"Rejected multi-element hyphenated soup: {name}"
    if len(clean_words.split()) > 5:
        return False, f"Rejected overly verbose composite name (>5 words): {name}"

    # 3. Reject known immiscible / chemically flawed products
    if "copper-lithium" in low_name or "lithium-copper" in low_name or "lithium copper" in low_name:
        return False, f"Rejected immiscible/non-standard Cu-Li construct: {name}"
    if "lithium sodium oxide" in low_name:
        return False, f"Rejected non-stoichiometric mixed alkali oxide: {name}"
    if "sulfur dichloride" in low_name and (bp is None or any(s.temperature_c > 150 for s in bp.steps)):
        return False, "Rejected Sulfur Dichloride at high temperature (decomposes >60°C)"
    if "nitride" in low_name:
        if bp is not None and any(s.atmosphere == "oxidizing_air" for s in bp.steps):
            return False, f"Rejected Nitride synthesis in oxidizing_air: {name}"
        if "combusting lithium in air" in low_desc or "in air" in low_desc:
            return False, f"Rejected Nitride formed in oxidizing air: {name}"

    # 4. Tier 4 and Tier 5 must be genuine functional components or mechanisms, never bulk ceramics/composites
    if tier >= 4 and category in ("ceramic", "composite", "compound", "alloy", "material", "glass", "salt"):
        return False, f"Rejected bulk material '{category}' masquerading as Tier {tier} component: {name}"

    return True, "OK"


class SharedLibrary:
    """Thread-safe shared item library with Strict Viability Filtering and Tool/Equipment Gating."""

    def __init__(
        self,
        state_path: str = "data/library_state.json",
        novel_registry_path: str = "data/novel_inventions_registry.json",
        reset_on_start: bool = False,
    ):
        self.lock = threading.RLock()
        self.state_path = state_path
        self.novel_registry_path = novel_registry_path

        self.items_by_id: Dict[str, LibraryItem] = {}
        self.name_to_id: Dict[str, str] = {}
        self.family_counts: Dict[str, int] = {}
        self.seen_blueprint_ids: Set[str] = set()
        self.experiences: List[ExperienceRecord] = []
        self.near_miss_queue: List[ExperienceRecord] = []
        self.novel_registry: Dict[str, Dict[str, Any]] = {}
        self.step_counter: int = 0
        self._simulator: Any = None

        for item in get_starting_primitives():
            item.novelty_classification = "SEED_PRIMITIVE"
            item.escalation_status = "NOT_REQUIRED"
            self.items_by_id[item.item_id] = item
            self.name_to_id[normalize_item_name(item.name)] = item.item_id

        if not reset_on_start and os.path.exists(self.state_path):
            self.load_state()
            self.purge_non_viable_items()
        else:
            self.save_state()

    def purge_non_viable_items(self) -> List[str]:
        """Remove any previously saved items that violate strict construct viability or depend on purged parents."""
        with self.lock:
            purged_names: List[str] = []
            changed = True
            while changed:
                changed = False
                for item_id in list(self.items_by_id.keys()):
                    it = self.items_by_id[item_id]
                    if it.discovered_by_worker == "seed_library":
                        continue
                    # Clean up parenthetical suffix in name if needed (e.g. "Calcium-Aluminum Alloy (Ca-Al)" -> "Calcium-Aluminum Alloy")
                    clean_name = re.sub(r"\s*\([^)]*\)\s*", " ", it.name).strip()
                    it.name = clean_name
                    if it.category in CATEGORY_TO_TIER:
                        it.tier = CATEGORY_TO_TIER[it.category]
                    if it.category in ("component", "mechanism", "device") and any(
                        pid in self.items_by_id and self.items_by_id[pid].tier >= 4
                        for pid in it.parent_ids
                    ):
                        it.category = "mechanism"
                        it.tier = 5

                    ok, _ = is_viable_construct(
                        name=it.name,
                        formula=it.formula_or_spec,
                        tier=it.tier,
                        category=it.category,
                        description=it.description,
                        bp=None,
                    )
                    missing_parent = any(
                        pid not in self.items_by_id for pid in it.parent_ids
                    )
                    if not ok or missing_parent:
                        purged_names.append(it.name)
                        del self.items_by_id[item_id]
                        changed = True

            # Rebuild indices, family counts, and real-world utility profiles
            self.name_to_id.clear()
            self.family_counts.clear()
            for it in self.items_by_id.values():
                self.name_to_id[normalize_item_name(it.name)] = it.item_id
                prof = derive_real_world_utility(
                    name=it.name,
                    formula=it.formula_or_spec,
                    tier=it.tier,
                    category=it.category,
                    capabilities=it.capabilities,
                    description=it.description,
                )
                if not it.real_world_equivalent or it.utility_score <= 0.80:
                    it.utility_score = max(it.utility_score, prof["utility_score"])
                    it.real_world_equivalent = prof["real_world_equivalent"]
                    it.practical_applications = prof["practical_applications"]
                    it.real_world_interpretation = prof["real_world_interpretation"]
                if it.item_id in self.novel_registry:
                    reg = self.novel_registry[it.item_id]
                    reg["tier"] = it.tier
                    reg["category"] = it.category
                    reg["utility_score"] = it.utility_score
                    reg["real_world_equivalent"] = it.real_world_equivalent
                    reg["practical_applications"] = it.practical_applications
                    reg["real_world_interpretation"] = it.real_world_interpretation
                    self._enrich_novel_flag_and_analysis(reg, it)
                if it.discovered_by_worker != "seed_library":
                    fam = extract_concept_family(it.name, it.category, it.capabilities)
                    self.family_counts[fam] = self.family_counts.get(fam, 0) + 1

            for reg_id, reg in self.novel_registry.items():
                if reg_id not in self.items_by_id:
                    self._enrich_novel_flag_and_analysis(reg, None)

            self.save_state()
            return purged_names

    def get_all_items(self, include_rejected_audits: bool = False) -> List[LibraryItem]:
        with self.lock:
            if include_rejected_audits:
                return list(self.items_by_id.values())
            return [
                it
                for it in self.items_by_id.values()
                if it.escalation_status != "REJECTED_DEEP_AUDIT"
            ]

    def get_item(self, item_id: str) -> Optional[LibraryItem]:
        with self.lock:
            return self.items_by_id.get(item_id)

    def get_seen_blueprint_ids(self) -> Set[str]:
        with self.lock:
            return set(self.seen_blueprint_ids)

    def get_item_family_saturation(self, item: LibraryItem) -> int:
        if item.discovered_by_worker == "seed_library":
            return 0
        fam = extract_concept_family(item.name, item.category, item.capabilities)
        with self.lock:
            return self.family_counts.get(fam, 0)

    def get_equipment_gates(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate which technological process gates have been unlocked by viable invented items."""
        with self.lock:
            invented = [
                it
                for it in self.items_by_id.values()
                if it.discovered_by_worker != "seed_library"
            ]

            def find_enabler(pred) -> Optional[str]:
                for it in invented:
                    if pred(it):
                        return f"{it.name} ({it.formula_or_spec})"
                return None

            wet_tool = find_enabler(
                lambda it: it.phase in ("liquid", "aqueous")
                or "water" in it.name.lower()
                or "acid" in it.name.lower()
                or "hydroxide" in it.name.lower()
            )
            high_temp_tool = find_enabler(
                lambda it: "refractory" in it.capabilities
                or "vessel" in it.capabilities
                or it.properties.get("thermal_stability", 0.0) >= 0.85
            )
            electro_tool = find_enabler(
                lambda it: "electrolyte" in it.capabilities
            )
            pressure_tool = find_enabler(
                lambda it: it.tier >= 3
                and (
                    "structural" in it.capabilities
                    or "vessel" in it.capabilities
                    or "sealant" in it.capabilities
                )
            )
            fab_tool = find_enabler(
                lambda it: it.tier >= 2
                and (
                    "insulator" in it.capabilities
                    or "conductor" in it.capabilities
                    or "semiconductor" in it.capabilities
                )
            )
            plasma_tool = find_enabler(
                lambda it: it.tier >= 4
                and it.category in ("component", "mechanism", "device")
                and (
                    "magnetic" in it.capabilities
                    or "energy_storage" in it.capabilities
                    or "heating_element" in it.capabilities
                    or "actuator" in it.capabilities
                )
            )

            return {
                "wet_chemistry": {
                    "unlocked": wet_tool is not None,
                    "equipment": wet_tool,
                    "description": "Unlocks dissolve, boil_reflux, distill_fractionate, crystallize_precipitate & aqueous_solution",
                },
                "high_temp_1500c": {
                    "unlocked": high_temp_tool is not None,
                    "equipment": high_temp_tool,
                    "description": "Unlocks 1500°C smelting (smelt_reduce, alloy_melt, sinter_fire at 1500°C)",
                },
                "electrochemistry": {
                    "unlocked": electro_tool is not None,
                    "equipment": electro_tool,
                    "description": "Unlocks electrolyze & coat_plate operations",
                },
                "high_pressure_50atm": {
                    "unlocked": pressure_tool is not None,
                    "equipment": pressure_tool,
                    "description": "Unlocks 50 atm & 500 atm autoclave/high-pressure synthesis",
                },
                "precision_fabrication": {
                    "unlocked": fab_tool is not None,
                    "equipment": fab_tool,
                    "description": "Unlocks wire_circuit, dope_diffuse, seal_encapsulate & polymerize",
                },
                "plasma_arc_5000c": {
                    "unlocked": plasma_tool is not None,
                    "equipment": plasma_tool,
                    "description": "Unlocks 5000°C plasma/arc/nuclear operations (ionize_bind, nucleosynthesis_fuse)",
                },
            }

    def mark_blueprint_claimed(self, blueprint_id: str) -> bool:
        with self.lock:
            if blueprint_id in self.seen_blueprint_ids:
                return False
            self.seen_blueprint_ids.add(blueprint_id)
            return True

    def pop_near_miss_for_repair(self) -> Optional[ExperienceRecord]:
        with self.lock:
            if not self.near_miss_queue:
                return None
            return self.near_miss_queue.pop(0)

    def _compute_child_lineage(self, bp: Blueprint) -> Dict[str, float]:
        lineage: Dict[str, float] = {}
        total_ratio = 0.0
        for ing in bp.ingredients:
            parent = self.items_by_id.get(ing.item_id)
            if not parent:
                continue
            r = max(1.0, float(ing.ratio))
            total_ratio += r
            for el, val in parent.elemental_lineage.items():
                lineage[el] = lineage.get(el, 0.0) + val * r
        if total_ratio > 0:
            for el in list(lineage.keys()):
                lineage[el] = round(lineage[el] / total_ratio, 4)
        return lineage

    def _find_semantic_duplicate(
        self,
        name: str,
        tier: int,
        category: str,
        capabilities: List[str],
        parent_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        norm = normalize_item_name(name)
        if norm in self.name_to_id:
            return self.name_to_id[norm]
        stop_words = {
            "assembled",
            "hybrid",
            "composite",
            "multi",
            "part",
            "with",
            "and",
            "the",
            "of",
            "mechanically",
            "system",
            "structure",
            "anhydrous",
            "alloy",
        }
        tokens = {t for t in norm.split("_") if len(t) > 2 and t not in stop_words}
        if not tokens:
            return None
        cap_set = set(capabilities)
        parent_set = set(parent_ids or [])
        for ex_id, ex_item in self.items_by_id.items():
            if ex_item.discovered_by_worker == "seed_library":
                continue
            if ex_id in parent_set:
                continue
            ex_tokens = {
                t
                for t in normalize_item_name(ex_item.name).split("_")
                if len(t) > 2 and t not in stop_words
            }
            if not ex_tokens:
                continue
            # Also catch reversed binary alloy names like "Copper-Lithium Alloy" vs "Lithium-Copper Alloy"
            if tokens == ex_tokens:
                return ex_id
            if ex_item.tier == tier:
                overlap = len(tokens & ex_tokens) / max(1, max(len(tokens), len(ex_tokens)))
                if overlap >= 0.75 and set(ex_item.capabilities) == cap_set:
                    return ex_id
        return None

    def record_evaluation(
        self, bp: Blueprint, verdict: JudgeVerdict
    ) -> Tuple[ExperienceRecord, Optional[LibraryItem]]:
        """Process a Judge verdict, enforce strict construct viability, compute reward, and save state."""
        with self.lock:
            self.step_counter += 1
            self.seen_blueprint_ids.add(bp.blueprint_id)

            # Enforce hard programmatic viability gate even if LLM returned valid=True
            if verdict.valid and verdict.output_name:
                clean_out_name = re.sub(r"\s*\([^)]*\)\s*", " ", verdict.output_name).strip()
                if "/" in clean_out_name and any(
                    k in clean_out_name.lower()
                    for k in ("cermet", "heterojunction", "junction", "diode", "cell", "motor", "solenoid")
                ):
                    clean_out_name = re.sub(r"\s*/\s*", "-", clean_out_name)
                verdict.output_name = clean_out_name
                viable, reason = is_viable_construct(
                    name=verdict.output_name,
                    formula=verdict.formula_or_spec,
                    tier=verdict.tier,
                    category=verdict.category,
                    description=verdict.description,
                    bp=bp,
                )
                if not viable:
                    verdict.valid = False
                    verdict.physics_score = min(verdict.physics_score, 0.25)
                    verdict.failure_reason = reason

            gates_before = {
                k: v["unlocked"] for k, v in self.get_equipment_gates().items()
            }
            all_caps_before = {
                c
                for it in self.items_by_id.values()
                for c in it.capabilities
                if it.discovered_by_worker != "seed_library"
            }

            parent_items = [
                self.items_by_id[ing.item_id]
                for ing in bp.ingredients
                if ing.item_id in self.items_by_id
            ]
            parent_caps = {c for it in parent_items for c in it.capabilities}

            is_new_discovery = False
            discovered_item: Optional[LibraryItem] = None
            discovered_item_id: Optional[str] = None

            if verdict.valid and verdict.output_name:
                if verdict.category in ("component", "mechanism", "device") and any(
                    pit.tier >= 4 for pit in parent_items
                ):
                    verdict.category = "mechanism"
                    verdict.tier = 5
                norm_name = normalize_item_name(verdict.output_name)
                existing_id = self._find_semantic_duplicate(
                    verdict.output_name,
                    verdict.tier,
                    verdict.category,
                    verdict.capabilities,
                    parent_ids=[ing.item_id for ing in bp.ingredients],
                )

                for pit in parent_items:
                    pit.times_used_in_valid_recipes += 1

                if existing_id is None:
                    is_new_discovery = True
                    new_id = f"inv_{verdict.tier}_{norm_name[:28]}_{len(self.items_by_id)}"
                    child_lineage = self._compute_child_lineage(bp)

                    is_novel = (
                        verdict.novelty_classification == "NOVEL_FRONTIER"
                        or not verdict.is_known_real_world
                    )
                    nov_class = "NOVEL_FRONTIER" if is_novel else "KNOWN_REAL_WORLD"
                    esc_status = "NOT_REQUIRED"
                    if is_novel:
                        sci_s = float((verdict.deep_audit or {}).get("science_math_score", 0.85))
                        nov_s = float((verdict.deep_audit or {}).get("novelty_score", 0.75))
                        util_s = float(verdict.utility_score)
                        if sci_s < 0.65:
                            esc_status = "REJECTED_DEEP_AUDIT"
                        elif nov_s < 0.50:
                            esc_status = "RECLASSIFIED_KNOWN"
                            is_novel = False
                            nov_class = "KNOWN_REAL_WORLD"
                        elif sci_s >= 0.80 and nov_s >= 0.55 and util_s >= 0.80:
                            esc_status = "VERIFIED_NOVEL_BREAKTHROUGH"
                        elif sci_s >= 0.72 and nov_s >= 0.50:
                            esc_status = "PLAUSIBLE_FRONTIER_CONSTRUCT"
                        else:
                            esc_status = "PENDING_ESCALATION"

                    discovered_item = LibraryItem(
                        item_id=new_id,
                        name=verdict.output_name,
                        formula_or_spec=verdict.formula_or_spec or verdict.output_name,
                        tier=verdict.tier,
                        category=verdict.category,
                        phase=verdict.phase,
                        description=verdict.description,
                        properties=verdict.properties,
                        capabilities=verdict.capabilities,
                        elemental_lineage=child_lineage,
                        parent_ids=[ing.item_id for ing in bp.ingredients],
                        created_by_blueprint_id=bp.blueprint_id,
                        discovered_by_worker=bp.proposer_id,
                        discovery_step=self.step_counter,
                        times_used_in_valid_recipes=0,
                        is_known_real_world=not is_novel,
                        novelty_classification=nov_class,
                        novelty_rationale=verdict.novelty_rationale,
                        escalation_status=esc_status,
                        recipe_summary=bp.summary_recipe_str(),
                        judge_votes=verdict.judge_votes,
                        deep_audit=verdict.deep_audit,
                        utility_score=verdict.utility_score,
                        real_world_equivalent=verdict.real_world_equivalent,
                        practical_applications=verdict.practical_applications,
                        real_world_interpretation=verdict.real_world_interpretation,
                    )
                    self.items_by_id[new_id] = discovered_item
                    self.name_to_id[norm_name] = new_id
                    discovered_item_id = new_id

                    if is_novel:
                        self.novel_registry[new_id] = {
                            "item_id": new_id,
                            "blueprint_id": bp.blueprint_id,
                            "name": discovered_item.name,
                            "formula_or_spec": discovered_item.formula_or_spec,
                            "tier": discovered_item.tier,
                            "category": discovered_item.category,
                            "phase": discovered_item.phase,
                            "description": discovered_item.description,
                            "properties": discovered_item.properties,
                            "capabilities": discovered_item.capabilities,
                            "discovered_by_worker": bp.proposer_id,
                            "strategy": bp.strategy,
                            "recipe_summary": bp.summary_recipe_str(),
                            "blueprint": bp.to_dict(),
                            "judge_votes": verdict.judge_votes,
                            "novelty_rationale": verdict.novelty_rationale,
                            "escalation_status": esc_status,
                            "deep_audit": verdict.deep_audit,
                            "utility_score": discovered_item.utility_score,
                            "real_world_equivalent": discovered_item.real_world_equivalent,
                            "practical_applications": discovered_item.practical_applications,
                            "real_world_interpretation": discovered_item.real_world_interpretation,
                            "discovered_at": time.time(),
                        }
                        self._enrich_novel_flag_and_analysis(self.novel_registry[new_id], discovered_item)

                    fam = extract_concept_family(
                        verdict.output_name, verdict.category, verdict.capabilities
                    )
                    prev_fam_count = self.family_counts.get(fam, 0)
                    self.family_counts[fam] = prev_fam_count + 1

                    gates_after = {
                        k: v["unlocked"] for k, v in self.get_equipment_gates().items()
                    }
                    newly_unlocked_gates = [
                        k for k, v in gates_after.items() if v and not gates_before.get(k)
                    ]
                    first_time_caps = [
                        c for c in verdict.capabilities if c not in all_caps_before
                    ]
                    emergent_caps = [
                        c for c in verdict.capabilities if c not in parent_caps
                    ]

                    novel_bonus = 6.0 if is_novel else 0.0
                    utility_bonus = 2.5 * float(verdict.utility_score) + (
                        2.0 if verdict.tier >= 4 and verdict.utility_score >= 0.85 else 0.0
                    )
                    base_reward = (
                        5.0
                        + 1.5 * float(verdict.tier)
                        + 0.8 * len(emergent_caps)
                        + 1.5 * len(first_time_caps)
                        + 4.0 * len(newly_unlocked_gates)
                        + novel_bonus
                        + utility_bonus
                        + 1.0 * float(verdict.physics_score)
                    )
                    saturation_mult = 0.45 ** max(0, prev_fam_count - 1)
                    reward = base_reward * saturation_mult
                else:
                    discovered_item_id = existing_id
                    ex_it = self.items_by_id.get(existing_id)
                    if ex_it and not ex_it.recipe_summary:
                        ex_it.recipe_summary = bp.summary_recipe_str()
                    if ex_it and not ex_it.judge_votes and verdict.judge_votes:
                        ex_it.judge_votes = verdict.judge_votes
                    reward = 0.4 + 0.3 * float(verdict.physics_score)
            else:
                reward = round(0.8 * float(verdict.physics_score) - 0.5, 4)

            exp = ExperienceRecord(
                blueprint=bp,
                verdict=verdict,
                reward=round(reward, 4),
                is_new_discovery=is_new_discovery,
                discovered_item_id=discovered_item_id,
                timestamp=time.time(),
            )
            self.experiences.append(exp)

            if not verdict.valid and (
                verdict.physics_score >= 0.35
                or bool(verdict.repair_hint.get("can_fix_recipe"))
            ):
                self.near_miss_queue.append(exp)
                if len(self.near_miss_queue) > 50:
                    self.near_miss_queue.pop(0)

            self.save_state()
            return exp, discovered_item

    def _enrich_novel_flag_and_analysis(
        self, reg: Dict[str, Any], it: Optional[LibraryItem] = None
    ) -> None:
        """Compute whether a novel invention is flagged as genuinely novel & practically useful, and attach a full 5-part analysis."""
        name = str(reg.get("name", ""))
        formula = str(reg.get("formula_or_spec", ""))
        tier = int(reg.get("tier", 3))
        cat = str(reg.get("category", "material"))
        caps = list(reg.get("capabilities", []))
        desc = str(reg.get("description", ""))

        prof = derive_real_world_utility(name, formula, tier, cat, caps, desc)
        util_score = float(reg.get("utility_score") or prof["utility_score"])
        reg["utility_score"] = max(util_score, prof["utility_score"])
        reg["real_world_equivalent"] = reg.get("real_world_equivalent") or prof["real_world_equivalent"]
        reg["practical_applications"] = reg.get("practical_applications") or prof["practical_applications"]
        reg["real_world_interpretation"] = reg.get("real_world_interpretation") or prof["real_world_interpretation"]

        da = reg.get("deep_audit") or {}
        if not isinstance(da, dict):
            da = {}
        sci_score = float(da.get("science_math_score", 0.85))
        nov_score = float(da.get("novelty_score", 0.78))
        esc = str(reg.get("escalation_status", "PENDING_ESCALATION"))

        if sci_score < 0.65 and esc not in ("REJECTED_DEEP_AUDIT", "RECLASSIFIED_KNOWN"):
            esc = "REJECTED_DEEP_AUDIT"
            reg["escalation_status"] = esc
            if it is not None:
                it.escalation_status = esc
        elif nov_score < 0.50 and sci_score >= 0.65 and esc not in ("REJECTED_DEEP_AUDIT", "RECLASSIFIED_KNOWN"):
            esc = "RECLASSIFIED_KNOWN"
            reg["escalation_status"] = esc
            da["chief_scientist_verdict"] = (
                f"RECLASSIFIED AS KNOWN REAL-WORLD CONSTRUCT (Science/Math={sci_score:.2f}, Novelty={nov_score:.2f}): "
                f"{name} ({formula}) is scientifically valid and useful ({reg['real_world_equivalent']}), "
                f"but is an established real-world chemical/material rather than an uncharted frontier invention."
            )
            reg["deep_audit"] = da
            if it is not None:
                it.escalation_status = esc
                it.novelty_classification = "KNOWN_REAL_WORLD"
                it.is_known_real_world = True
                it.deep_audit = da

        is_flagged = (
            esc in ("VERIFIED_NOVEL_BREAKTHROUGH", "PLAUSIBLE_FRONTIER_CONSTRUCT")
            and sci_score >= 0.78
            and nov_score >= 0.55
            and reg["utility_score"] >= 0.80
        )
        reg["flagged_novel_useful"] = is_flagged

        why_novel = (
            str(da.get("novelty_vs_known_world", "")).strip()
            or str(reg.get("novelty_rationale", "")).strip()
            or f"{name} ({formula}) combines non-standard phase chemistry or functional component coupling that is absent from standard commercial catalogs."
        )
        why_sci = (
            str(da.get("thermodynamics_and_math", "")).strip()
            or "Thermodynamic phase stability, stoichiometry, and processing conditions verified by 3-model consensus and first-principles audit."
        )
        mech = (
            str(da.get("mechanism_and_structure", "")).strip()
            or desc
        )
        apps_list = reg.get("practical_applications") or da.get("potential_applications") or []
        apps_str = "; ".join(str(a) for a in apps_list[:4])

        if is_flagged and (not da.get("chief_scientist_verdict") or "Pending" in str(da.get("chief_scientist_verdict", ""))):
            da["chief_scientist_verdict"] = (
                f"FLAGGED AS GENUINELY NOVEL & HIGH-UTILITY BREAKTHROUGH (Science/Math={sci_score:.2f}, Utility={reg['utility_score']:.2f}): "
                f"{name} ({formula}) is thermodynamically and mechanically sound while offering direct real-world utility as a "
                f"{reg['real_world_equivalent']}. {reg['real_world_interpretation']}"
            )
            da["reviewed_by"] = "Antigravity Autonomous Chief Scientist Audit"
            da["reviewed_at"] = time.time()
            reg["deep_audit"] = da
            if it is not None:
                it.deep_audit = da
                it.escalation_status = esc

        reg["novel_utility_analysis"] = {
            "flagged": is_flagged,
            "headline": (
                f"🚨 FLAGGED NOVEL & USEFUL BREAKTHROUGH: {name} ({formula}) → {reg['real_world_equivalent']}"
                if is_flagged
                else f"Audit Dossier: {name} ({formula})"
            ),
            "why_genuinely_novel": why_novel,
            "why_science_and_math_work": f"{why_sci} | Atomic/Engineering Mechanism: {mech}",
            "why_useful_in_real_world": (
                f"Translates directly into a {reg['real_world_equivalent']} (Utility Score: {int(round(reg['utility_score']*100))}%). "
                f"{reg['real_world_interpretation']} Key real-world applications: {apps_str}."
            ),
            "deployment_and_manufacturing_spec": (
                f"Synthesis Blueprint: {reg.get('recipe_summary', 'N/A')}. "
                f"Constraints: {da.get('synthesis_risks_and_constraints', 'Maintain controlled atmosphere and thermal profile.')}"
            ),
        }

        if is_flagged and not reg.get("simulation_verification"):
            try:
                sim_res = self._get_simulator().simulate_item(reg)
                reg["simulation_verification"] = sim_res
                if it is not None:
                    it.simulation_verification = sim_res
            except Exception:
                pass
        elif reg.get("simulation_verification") and it is not None and not it.simulation_verification:
            it.simulation_verification = reg["simulation_verification"]

    def _get_simulator(self) -> Any:
        if self._simulator is None:
            from inventor.simulator import PhysicsVerificationSimulator
            self._simulator = PhysicsVerificationSimulator()
        return self._simulator

    def run_simulation_for_item(self, item_id: str) -> Dict[str, Any]:
        """Run GPU Atomistic & Multiphysics simulation for any library or novel registry item and persist the result."""
        with self.lock:
            it = self.items_by_id.get(item_id)
            reg = self.novel_registry.get(item_id)
            target_dict = reg if reg is not None else (it.to_dict() if it is not None else None)
            if target_dict is None:
                return {}
        sim_res = self._get_simulator().simulate_item(target_dict)
        with self.lock:
            if it is not None:
                it.simulation_verification = sim_res
            if reg is not None:
                reg["simulation_verification"] = sim_res
            self.save_state()
        return sim_res

    def get_novel_inventions(self) -> List[Dict[str, Any]]:
        """Return all novel frontier inventions sorted newest-first."""
        with self.lock:
            out = list(self.novel_registry.values())
            out.sort(key=lambda x: float(x.get("discovered_at", 0.0)), reverse=True)
            return out

    def update_escalation_review(
        self,
        item_id: str,
        escalation_status: str,
        chief_scientist_verdict: str,
        reviewed_by: str = "Antigravity (Chief Scientist Escalation)",
        science_math_score: Optional[float] = None,
        novelty_score: Optional[float] = None,
    ) -> bool:
        """Update the escalation review status and Chief Scientist verdict for an invention."""
        with self.lock:
            it = self.items_by_id.get(item_id)
            reg = self.novel_registry.get(item_id)
            if not it and not reg:
                return False

            if it:
                it.escalation_status = escalation_status
                if escalation_status == "RECLASSIFIED_KNOWN":
                    it.is_known_real_world = True
                    it.novelty_classification = "KNOWN_REAL_WORLD"
                else:
                    it.is_known_real_world = False
                    it.novelty_classification = "NOVEL_FRONTIER"
                if not it.deep_audit:
                    it.deep_audit = {}
                it.deep_audit["chief_scientist_verdict"] = chief_scientist_verdict
                it.deep_audit["reviewed_by"] = reviewed_by
                it.deep_audit["reviewed_at"] = time.time()
                if science_math_score is not None:
                    it.deep_audit["science_math_score"] = round(float(science_math_score), 3)
                if novelty_score is not None:
                    it.deep_audit["novelty_score"] = round(float(novelty_score), 3)

            if reg:
                reg["escalation_status"] = escalation_status
                if "deep_audit" not in reg or not isinstance(reg["deep_audit"], dict):
                    reg["deep_audit"] = {}
                reg["deep_audit"]["chief_scientist_verdict"] = chief_scientist_verdict
                reg["deep_audit"]["reviewed_by"] = reviewed_by
                reg["deep_audit"]["reviewed_at"] = time.time()
                if science_math_score is not None:
                    reg["deep_audit"]["science_math_score"] = round(float(science_math_score), 3)
                if novelty_score is not None:
                    reg["deep_audit"]["novelty_score"] = round(float(novelty_score), 3)
            elif it:
                self.novel_registry[item_id] = {
                    "item_id": it.item_id,
                    "blueprint_id": it.created_by_blueprint_id or "",
                    "name": it.name,
                    "formula_or_spec": it.formula_or_spec,
                    "tier": it.tier,
                    "category": it.category,
                    "phase": it.phase,
                    "description": it.description,
                    "properties": it.properties,
                    "capabilities": it.capabilities,
                    "discovered_by_worker": it.discovered_by_worker,
                    "strategy": "escalated_review",
                    "recipe_summary": it.recipe_summary,
                    "blueprint": {},
                    "judge_votes": it.judge_votes,
                    "novelty_rationale": it.novelty_rationale,
                    "escalation_status": escalation_status,
                    "deep_audit": it.deep_audit,
                    "discovered_at": time.time(),
                }

            if item_id in self.novel_registry:
                self._enrich_novel_flag_and_analysis(self.novel_registry[item_id], it)

            self.save_state()
            return True

    def get_summary_stats(self) -> Dict[str, Any]:
        with self.lock:
            tier_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            seed_count = 0
            invented_count = 0
            known_invented_count = 0
            novel_frontier_count = 0
            pending_escalation_count = 0
            verified_novel_count = 0
            all_caps: Set[str] = set()
            for it in self.items_by_id.values():
                tier_counts[it.tier] = tier_counts.get(it.tier, 0) + 1
                all_caps.update(it.capabilities)
                if it.discovered_by_worker == "seed_library":
                    seed_count += 1
                else:
                    invented_count += 1
                    if it.novelty_classification == "NOVEL_FRONTIER" or not it.is_known_real_world:
                        novel_frontier_count += 1
                        if it.escalation_status == "PENDING_ESCALATION":
                            pending_escalation_count += 1
                        elif it.escalation_status in ("VERIFIED_NOVEL_BREAKTHROUGH", "PLAUSIBLE_FRONTIER_CONSTRUCT"):
                            verified_novel_count += 1
                    else:
                        known_invented_count += 1

            total_evals = len(self.experiences)
            valid_evals = sum(1 for e in self.experiences if e.verdict.valid)
            new_discoveries = sum(1 for e in self.experiences if e.is_new_discovery)
            gates = self.get_equipment_gates()
            unlocked_gate_names = [k for k, v in gates.items() if v["unlocked"]]

            return {
                "total_library_items": len(self.items_by_id),
                "seed_primitives": seed_count,
                "invented_items": invented_count,
                "known_real_world_items": known_invented_count,
                "novel_frontier_items": novel_frontier_count,
                "pending_escalation_reviews": pending_escalation_count,
                "verified_novel_breakthroughs": verified_novel_count,
                "tier_counts": tier_counts,
                "capabilities_unlocked": sorted(all_caps),
                "equipment_gates_unlocked": f"{len(unlocked_gate_names)}/{len(gates)}",
                "equipment_gates": gates,
                "concept_families": dict(self.family_counts),
                "total_evaluations": total_evals,
                "valid_blueprints": valid_evals,
                "new_discoveries": new_discoveries,
                "hit_rate": round(valid_evals / max(1, total_evals), 3),
                "near_miss_queue_size": len(self.near_miss_queue),
            }

    def _atomic_write_json(self, target_path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        tmp_path = f"{target_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        for attempt in range(5):
            try:
                os.replace(tmp_path, target_path)
                return
            except OSError:
                time.sleep(0.05 * (attempt + 1))
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError:
            pass
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save_state(self) -> None:
        with self.lock:
            data = {
                "step_counter": self.step_counter,
                "stats": self.get_summary_stats(),
                "items": [it.to_dict() for it in self.items_by_id.values()],
                "recent_experiences": [
                    e.to_dict() for e in self.experiences[-250:]
                ],
            }
            self._atomic_write_json(self.state_path, data)

            nov_data = {
                "updated_at": time.time(),
                "total_novel_inventions": len(self.novel_registry),
                "novel_inventions": self.get_novel_inventions(),
            }
            self._atomic_write_json(self.novel_registry_path, nov_data)

    def load_state(self) -> None:
        try:
            with self.lock:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.step_counter = int(data.get("step_counter", 0))
                self.family_counts.clear()
                for raw_it in data.get("items", []):
                    it = LibraryItem.from_dict(raw_it)
                    self.items_by_id[it.item_id] = it
                    self.name_to_id[normalize_item_name(it.name)] = it.item_id
                    if it.created_by_blueprint_id:
                        self.seen_blueprint_ids.add(it.created_by_blueprint_id)
                if not self.experiences and data.get("recent_experiences"):
                    for raw_exp in data.get("recent_experiences", []):
                        try:
                            exp = ExperienceRecord.from_dict(raw_exp)
                            self.experiences.append(exp)
                            self.seen_blueprint_ids.add(exp.blueprint.blueprint_id)
                            # Backfill recipe_summary and judge_votes on items if missing
                            if exp.discovered_item_id and exp.discovered_item_id in self.items_by_id:
                                d_it = self.items_by_id[exp.discovered_item_id]
                                if not d_it.recipe_summary:
                                    d_it.recipe_summary = exp.blueprint.summary_recipe_str()
                                if not d_it.judge_votes and exp.verdict.judge_votes:
                                    d_it.judge_votes = exp.verdict.judge_votes
                        except Exception:
                            pass
                if os.path.exists(self.novel_registry_path):
                    with open(self.novel_registry_path, "r", encoding="utf-8") as nf:
                        ndata = json.load(nf)
                    for entry in ndata.get("novel_inventions", []):
                        iid = entry.get("item_id")
                        if iid and iid in self.items_by_id:
                            self.novel_registry[iid] = entry
        except Exception:
            pass

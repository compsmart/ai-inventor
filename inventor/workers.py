"""Parallel Multi-Inventor Swarm Workers with Strict Viability & Clean Engineering Progression."""

from __future__ import annotations

import concurrent.futures
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from inventor.judge import DeepSeekJudge
from inventor.library import SharedLibrary
from inventor.neural_net import SharedBrain
from inventor.primitives import DESIGN_GOALS
from inventor.schemas import (
    ATMOS_TO_IDX,
    OP_TO_IDX,
    Blueprint,
    BlueprintIngredient,
    ExperienceRecord,
    LibraryItem,
    ProcessStep,
)


class InventorWorker:
    """An individual autonomous Inventor agent proposing clean chemical, metallurgical, and engineering blueprints."""

    def __init__(
        self,
        worker_id: str,
        role_name: str,
        library: SharedLibrary,
        brain: SharedBrain,
        judge: DeepSeekJudge,
    ):
        self.worker_id = worker_id
        self.role_name = role_name
        self.library = library
        self.brain = brain
        self.judge = judge

    def _build_synergy_subsets(
        self,
        items: List[LibraryItem],
        mode: str,
        equipment_gates: Dict[str, Dict[str, Any]],
        count: int = 96,
    ) -> List[List[LibraryItem]]:
        """Generate candidate ingredient subsets focused on viable, standard real-world constructs."""
        subsets: List[List[LibraryItem]] = []
        if not items:
            return subsets

        fam_counts = self.library.family_counts
        oxides_saturated = fam_counts.get("oxide_compound", 0) >= 5
        chlorides_saturated = fam_counts.get("chloride_compound", 0) >= 5

        by_cap: Dict[str, List[LibraryItem]] = {}
        by_tier: Dict[int, List[LibraryItem]] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        by_id: Dict[str, LibraryItem] = {}
        existing_parent_sets: Set[Tuple[str, ...]] = set()
        for it in items:
            by_id[it.item_id] = it
            by_tier.setdefault(it.tier, []).append(it)
            if it.parent_ids:
                existing_parent_sets.add(tuple(sorted(it.parent_ids)))
            if it.tier >= 1:
                for c in it.capabilities:
                    by_cap.setdefault(c, []).append(it)

        def pick(pool: List[LibraryItem], boost_higher_tier: bool = True) -> Optional[LibraryItem]:
            if not pool:
                return None
            weights = []
            for x in pool:
                tier_boost = (1.0 + 0.8 * x.tier) if boost_higher_tier else 1.0
                usage_pen = 1.0 + 1.2 * (x.times_used_in_valid_recipes ** 2)
                elem_pen = 1.0
                if oxides_saturated and x.item_id == "elem_o":
                    elem_pen = 20.0
                if chlorides_saturated and x.item_id == "elem_cl":
                    elem_pen = 20.0
                w = tier_boost / (usage_pen * elem_pen)
                weights.append(max(1e-5, w))
            return random.choices(pool, weights=weights, k=1)[0]

        wet_unlocked = equipment_gates.get("wet_chemistry", {}).get("unlocked", False)
        ht_unlocked = equipment_gates.get("high_temp_1500c", {}).get("unlocked", False)
        elec_unlocked = equipment_gates.get("electrochemistry", {}).get("unlocked", False)
        plasma_unlocked = equipment_gates.get("plasma_arc_5000c", {}).get("unlocked", False)

        pure_metals = [
            by_id[k]
            for k in ("elem_fe", "elem_cu", "elem_al", "elem_ca", "elem_na", "elem_li")
            if k in by_id
        ]
        pure_metalloids = [
            by_id[k] for k in ("elem_c", "elem_si") if k in by_id
        ]
        binary_oxides = [
            it
            for it in by_tier[2]
            if it.name
            in (
                "Silicon Dioxide",
                "Aluminum Oxide",
                "Calcium Oxide",
                "Sodium Oxide",
                "Lithium Oxide",
                "Iron Oxide",
                "Iron(III) Oxide",
                "Copper Oxide",
                "Copper(II) Oxide",
            )
        ]
        salts_and_electrolytes = [
            it
            for it in (by_tier[2] + by_tier[3])
            if "electrolyte" in it.capabilities or it.category in ("salt", "acid", "base")
        ]

        for _ in range(count):
            chosen: List[LibraryItem] = []

            # Priority Bootstrap for locked gates
            if not wet_unlocked and random.random() < 0.55:
                # Synthesize Water (2 H2 + O2 -> 2 H2O) to unlock Wet Chemistry gate
                h = by_id.get("elem_h")
                o = by_id.get("elem_o")
                if h and o and random.random() < 0.75:
                    subsets.append([h, o])
                    continue
                partner = random.choice(
                    [by_id[k] for k in ("elem_o", "elem_cl", "elem_s") if k in by_id]
                )
                if h and partner:
                    subsets.append([h, partner])
                    continue

            if not ht_unlocked and random.random() < 0.28:
                ox = by_id.get("elem_o")
                refr = [by_id[k] for k in ("elem_al", "elem_si", "elem_ca") if k in by_id]
                if ox and refr:
                    subsets.append([random.choice(refr), ox])
                    continue

            if mode == "chemist":
                r = random.random()
                water_it = next((x for x in items if x.name == "Water"), None)
                photon_it = by_id.get("sub_photon")
                fe2o3_it = next((x for x in items if "iron oxide" in x.name.lower() or x.name == "Silicon Carbide"), None)
                if r < 0.08 and plasma_unlocked and by_id.get("sub_proton") and by_id.get("sub_electron"):
                    if by_id.get("sub_neutron") and random.random() < 0.5:
                        chosen = [by_id["sub_proton"], by_id["sub_neutron"], by_id["sub_electron"]]
                    else:
                        chosen = [by_id["sub_proton"], by_id["sub_electron"]]
                elif r < 0.24 and wet_unlocked and water_it and photon_it and fe2o3_it:
                    # Photoelectrochemical Solar Water Splitting (Hematite Fe2O3 + Water + Photon hν)
                    chosen = [fe2o3_it, water_it, photon_it]
                elif r < 0.55 and by_id.get("elem_h"):
                    # Hydrogen reactions: Ammonia (3H + N + Fe catalyst), Methane (C + 2H + Fe catalyst), HCl (H+Cl), H2S (H+S), Water (2H+O)
                    h = by_id["elem_h"]
                    partners = [
                        by_id[k]
                        for k in ("elem_n", "elem_c", "elem_cl", "elem_s", "elem_o")
                        if k in by_id
                    ]
                    p = pick(partners, boost_higher_tier=False)
                    if p:
                        if p.item_id in ("elem_n", "elem_c") and by_id.get("elem_fe"):
                            chosen = [h, p, by_id["elem_fe"]]
                        else:
                            chosen = [h, p]
                elif r < 0.85 and wet_unlocked:
                    # Aqueous chemistry: Oxide / Salt + Water -> Hydroxide / Acid / Brine Electrolyte (Na2O+H2O -> NaOH, Li2O+H2O -> LiOH, NaCl+H2O -> Brine)
                    solutes = [
                        x for x in (binary_oxides + by_tier[2])
                        if x.name in ("Sodium Oxide", "Lithium Oxide", "Calcium Oxide", "Sodium Chloride", "Lithium Chloride", "Hydrogen Chloride", "Sulfur Dioxide")
                    ] + [by_id[k] for k in ("elem_na", "elem_ca", "elem_li") if k in by_id]
                    solute = pick(solutes, boost_higher_tier=False)
                    if water_it and solute and water_it.item_id != solute.item_id:
                        chosen = [solute, water_it]
                else:
                    # Clean two-element reactions (e.g. S + O2 -> SO2, C + S -> CS2, Si + C -> SiC)
                    if by_id.get("elem_s") and by_id.get("elem_o") and random.random() < 0.55:
                        chosen = [by_id["elem_s"], by_id["elem_o"]]
                    else:
                        el_pool = [
                            x
                            for x in by_tier[1]
                            if not (oxides_saturated and x.item_id == "elem_o")
                            and not (chlorides_saturated and x.item_id == "elem_cl")
                            and "deuterium" not in x.name.lower()
                            and "atomic hydrogen" not in x.name.lower()
                        ]
                        if len(el_pool) >= 2:
                            chosen = random.sample(el_pool, 2)

            elif mode == "metallurgist":
                r = random.random()
                water_it = next((x for x in items if x.name == "Water"), None)
                ca_cement = next((x for x in items if "calcium aluminate" in x.name.lower()), None)
                ca_hydroxide = next((x for x in items if "calcium hydroxide" in x.name.lower()), None)
                sio2 = next((x for x in binary_oxides if x.name == "Silicon Dioxide"), None)
                si_it = by_id.get("elem_si")
                n0_it = by_id.get("sub_neutron")
                al_it = by_id.get("elem_al")
                volt_it = by_id.get("sub_voltage")

                if r < 0.16 and plasma_unlocked and si_it and n0_it:
                    # Neutron Transmutation Doping of Silicon (30Si + n0 -> 31P N-Type Silicon)
                    chosen = [si_it, n0_it]
                elif r < 0.30 and elec_unlocked and wet_unlocked and al_it and water_it and volt_it:
                    # Anodic Oxidation (Anodized Aluminum Dielectric Substrate via DC Voltage Bias)
                    chosen = [al_it, water_it, volt_it]
                elif r < 0.42 and wet_unlocked and water_it and (ca_cement or (ca_hydroxide and sio2)):
                    # Hydraulic Concrete / Pozzolanic Mortar Hydration
                    if ca_cement and random.random() < 0.55:
                        chosen = [ca_cement, water_it]
                    elif ca_hydroxide and sio2:
                        chosen = [ca_hydroxide, sio2, water_it]
                elif r < 0.58 and len(pure_metals) >= 1:
                    # Clean binary alloys & carbides: Cu+Al (Aluminum Bronze), Cu+Si (Silicon Bronze), Fe+C, Fe+Si, Si+C, Al+Si
                    pairs = [
                        ("elem_cu", "elem_si"),
                        ("elem_cu", "elem_al"),
                        ("elem_fe", "elem_c"),
                        ("elem_fe", "elem_si"),
                        ("elem_si", "elem_c"),
                        ("elem_al", "elem_si"),
                    ]
                    unexplored_pairs = [
                        (a, b)
                        for (a, b) in pairs
                        if a in by_id and b in by_id and tuple(sorted([a, b])) not in existing_parent_sets
                    ]
                    a_id, b_id = random.choice(unexplored_pairs if unexplored_pairs else pairs)
                    if a_id in by_id and b_id in by_id:
                        chosen = [by_id[a_id], by_id[b_id]]
                elif r < 0.74 and len(binary_oxides) >= 2:
                    # Clean industrial glasses & cements from binary oxides (e.g. SiO2 + CaO, SiO2 + Na2O + CaO -> Soda-Lime Glass, CaO + SiO2 + Al2O3 -> Portland Cement Clinker)
                    others = [x for x in binary_oxides if x.name in ("Calcium Oxide", "Sodium Oxide", "Lithium Oxide", "Aluminum Oxide")]
                    if sio2 and others:
                        if len(others) >= 2 and random.random() < 0.50:
                            chosen = [sio2] + random.sample(others, 2)
                        else:
                            chosen = [sio2, pick(others)]
                elif r < 0.90 and by_tier[3]:
                    # Solid-State Cermet Exploration (High-Melting Iron/Silicide/Aluminide/Aluminum Matrix + SiC or Al2O3 Ceramic Reinforcement)
                    matrices = [
                        x for x in by_tier[3]
                        if (x.category == "alloy" or "aluminide" in x.name.lower() or "silicide" in x.name.lower())
                        and "copper" not in x.name.lower()
                        and "bronze" not in x.name.lower()
                    ]
                    reinforcements = [
                        x for x in (by_tier[2] + by_tier[3])
                        if x.name in ("Silicon Carbide", "Aluminum Oxide")
                    ]
                    mat = pick(matrices if matrices else by_tier[3])
                    reinf = pick([x for x in reinforcements if mat and x.item_id != mat.item_id], boost_higher_tier=False)
                    if mat and reinf:
                        chosen = [mat, reinf]
                else:
                    # Single-material shaping/sintering: Copper -> Drawn Copper Wire, Carbon -> Graphite Electrode, Al2O3 -> Refractory Crucible
                    single_pool = [
                        by_id[k] for k in ("elem_cu", "elem_c", "elem_fe", "elem_si") if k in by_id
                    ] + [x for x in binary_oxides if x.name in ("Aluminum Oxide", "Silicon Dioxide")]
                    s = pick(single_pool)
                    if s:
                        chosen = [s]

            elif mode == "engineer":
                r = random.random()
                cu_it = by_id.get("elem_cu")
                al_it = by_id.get("elem_al")
                al2o3_it = next((x for x in items if x.name == "Aluminum Oxide"), None)
                sio2_it = next((x for x in items if x.name == "Silicon Dioxide"), None)
                sic_it = next((x for x in items if x.name == "Silicon Carbide"), None)
                water_it = next((x for x in items if x.name == "Water"), None)
                p_wafer = next((x for x in items if ("p-type" in x.name.lower() or "doped silicon wafer" in x.name.lower()) and "phosphorus" not in x.name.lower()), None)
                n_wafer = next((x for x in items if "phosphorus doped silicon" in x.name.lower() or "n-type" in x.name.lower()), None)
                hetero_it = next((x for x in items if "heterojunction" in x.name.lower()), None)
                solenoid_it = next((x for x in items if "solenoid" in x.name.lower()), None)
                heater_it = next((x for x in items if "heating element" in x.name.lower()), None)
                si_steel = next((x for x in items if x.name in ("Silicon Steel", "Ferrosilicon")), None)
                c_steel = next((x for x in items if x.name == "Carbon Steel"), None)
                al_bronze = next((x for x in items if "bronze" in x.name.lower()), None)
                graphite_it = next((x for x in items if "graphite" in x.name.lower()), by_id.get("elem_c"))
                photon_it = by_id.get("sub_photon")
                volt_it = by_id.get("sub_voltage")
                elec_it = by_id.get("sub_electron")

                anodes = [by_id[k] for k in ("elem_li", "elem_al", "elem_ca", "elem_na") if k in by_id]
                cathodes = [x for x in items if x.name in ("Synthetic Graphite", "Carbon", "Iron Oxide", "Copper Oxide", "Iron Disulfide")]
                electrolytes = [
                    x for x in items
                    if x.name in (
                        "Sodium Chloride",
                        "Lithium Chloride",
                        "Calcium Chloride",
                        "Calcium Hydroxide",
                        "Sodium Hydroxide",
                        "Lithium Aluminate Silicate",
                        "Lithium Aluminum Silicate",
                        "Lithium Silicate",
                        "Sodium Beta-Ferrite",
                    )
                    or "electrolyte" in x.capabilities
                ]

                if r < 0.24 and photon_it and (hetero_it or (p_wafer and n_wafer)):
                    # Archetype P1: Photovoltaic Solar Cell / Optical Photodiode (PN-Junction or Heterojunction + Photon hν)
                    if p_wafer and n_wafer and random.random() < 0.65:
                        chosen = [p_wafer, n_wafer, photon_it]
                    elif hetero_it and cu_it:
                        chosen = [hetero_it, cu_it, photon_it]
                    elif hetero_it:
                        chosen = [hetero_it, photon_it]
                elif r < 0.30 and volt_it and sic_it and (al_it or cu_it):
                    # Archetype P2: Silicon Carbide Light-Emitting Diode (SiC + Metal Contact + Electric Potential V_DC)
                    contact = al_it if (al_it and random.random() < 0.6) else (cu_it or al_it)
                    chosen = [sic_it, contact, volt_it]
                elif r < 0.42 and volt_it and solenoid_it and c_steel:
                    # Archetype P3: Magnetically Poled Permanent Magnet (Carbon Steel + Electromagnet Solenoid + V_DC)
                    chosen = [c_steel, solenoid_it, volt_it]
                elif r < 0.52 and volt_it and elec_it and graphite_it:
                    # Archetype P4: Thermionic Vacuum Electron Emitter (Graphite + Electron + High-Voltage V_DC)
                    chosen = [graphite_it, elec_it, volt_it]
                elif r < 0.62 and p_wafer and n_wafer:
                    # Archetype A: Silicon PN-Junction Diode / Photovoltaic Solar Cell
                    chosen = [p_wafer, n_wafer, cu_it] if (cu_it and random.random() < 0.5) else [p_wafer, n_wafer]
                elif r < 0.72 and (p_wafer or n_wafer) and (sio2_it or al2o3_it) and (al_it or cu_it):
                    # Archetype B: MOS Capacitor / Gate Stack
                    wafer = p_wafer if (p_wafer and random.random() < 0.6) else (n_wafer or p_wafer)
                    diel = sio2_it if (sio2_it and random.random() < 0.6) else (al2o3_it or sio2_it)
                    gate = al_it if (al_it and random.random() < 0.6) else (cu_it or al_it)
                    if wafer and diel and gate:
                        chosen = [wafer, diel, gate]
                elif r < 0.80 and anodes and cathodes and electrolytes:
                    # Archetype C: Galvanic / Voltaic Battery Cell
                    el = pick(electrolytes, boost_higher_tier=False)
                    if el and "hydroxide" in el.name.lower() and al_it:
                        an = al_it
                    else:
                        an = pick(anodes, boost_higher_tier=False)
                    ca = pick([x for x in cathodes if an and x.item_id != an.item_id], boost_higher_tier=False)
                    if an and ca and el:
                        chosen = [an, ca, el]
                elif r < 0.88 and heater_it and al2o3_it and cu_it:
                    # Archetype F: High-Temperature Electric Resistance Furnace
                    ca_cem = next((x for x in items if "calcium aluminate" in x.name.lower()), None)
                    refr = ca_cem if (ca_cem and random.random() < 0.5) else al2o3_it
                    chosen = [heater_it, refr, cu_it]
                elif r < 0.94 and solenoid_it and (si_steel or c_steel):
                    # Archetype E: Electrical Step-Up/Step-Down Transformer OR Electromagnetic Motor/Relay
                    if si_steel and random.random() < 0.55:
                        chosen = [solenoid_it, si_steel]
                    elif c_steel and al_bronze:
                        chosen = [solenoid_it, c_steel, al_bronze]
                    elif c_steel:
                        chosen = [solenoid_it, c_steel]
                elif by_id.get("elem_si"):
                    si = by_id["elem_si"]
                    dopants = [by_id[k] for k in ("elem_al", "elem_li") if k in by_id]
                    dop = pick(dopants, boost_higher_tier=False)
                    if dop:
                        chosen = [si, dop]

            if not chosen:
                pool = pure_metals + pure_metalloids + binary_oxides
                if len(pool) >= 2:
                    chosen = random.sample(pool, 2)
                else:
                    chosen = random.sample(items, min(2, len(items)))

            # If this parent combination has already been discovered, pivot 85% of the time to an unexplored frontier combination
            chosen_key = tuple(sorted(x.item_id for x in chosen))
            if chosen_key in existing_parent_sets and random.random() < 0.85:
                field_prims = [by_id[k] for k in ("sub_photon", "sub_voltage", "sub_neutron", "sub_electron") if k in by_id]
                if mode == "engineer":
                    eng_pool = [
                        x for x in items
                        if x.tier in (2, 3, 4)
                        and any(c in x.capabilities for c in ("semiconductor", "electrolyte", "conductor", "magnetic", "actuator", "sensor", "optical", "heating_element", "insulator", "refractory"))
                    ]
                    for _try in range(25):
                        if eng_pool and len(eng_pool) >= 2:
                            if field_prims and random.random() < 0.35:
                                cand = random.sample(eng_pool, 2) + [random.choice(field_prims)]
                            elif len(eng_pool) >= 3 and random.random() < 0.45:
                                cand = random.sample(eng_pool, 3)
                            else:
                                cand = random.sample(eng_pool, 2)
                            cand_key = tuple(sorted(x.item_id for x in cand))
                            if len(set(x.item_id for x in cand)) == len(cand) and cand_key not in existing_parent_sets:
                                chosen = cand
                                existing_parent_sets.add(cand_key)
                                break
                elif mode == "metallurgist":
                    met_pool = [
                        x for x in items
                        if x.tier in (1, 2, 3)
                        and x.category in ("element", "compound", "alloy", "ceramic", "material", "cermet", "intermetallic")
                        and "ingot" not in x.name.lower()
                        and "billet" not in x.name.lower()
                        and "wrought" not in x.name.lower()
                    ]
                    for _try in range(25):
                        if len(met_pool) >= 2:
                            cand = random.sample(met_pool, 2)
                            if field_prims and random.random() < 0.20:
                                cand = cand + [random.choice(field_prims)]
                            cand_key = tuple(sorted(x.item_id for x in cand))
                            if len(set(x.item_id for x in cand)) == len(cand) and cand_key not in existing_parent_sets:
                                chosen = cand
                                existing_parent_sets.add(cand_key)
                                break
                else:
                    chem_pool = [
                        x for x in items
                        if x.tier in (1, 2)
                        and x.category in ("element", "compound", "molecule", "salt", "acid", "base")
                    ]
                    for _try in range(25):
                        if len(chem_pool) >= 2:
                            cand = random.sample(chem_pool, 2 if random.random() < 0.7 else min(3, len(chem_pool)))
                            cand_key = tuple(sorted(x.item_id for x in cand))
                            if len(set(x.item_id for x in cand)) == len(cand) and cand_key not in existing_parent_sets:
                                chosen = cand
                                existing_parent_sets.add(cand_key)
                                break

            subsets.append(chosen)

        return subsets

    def _propose_from_critique_repair(
        self, equipment_gates: Dict[str, Dict[str, Any]]
    ) -> Optional[Blueprint]:
        """Worker 4 Strategy: Take a near-miss failed blueprint and repair its recipe within unlocked gates."""
        near_miss = self.library.pop_near_miss_for_repair()
        if near_miss is None:
            return None

        old_bp = near_miss.blueprint
        # Skip repairing blueprints whose ingredients were purged
        if any(self.library.get_item(ing.item_id) is None for ing in old_bp.ingredients):
            return None

        hint = near_miss.verdict.repair_hint or {}
        seen = self.library.get_seen_blueprint_ids()
        ht_unlocked = equipment_gates.get("high_temp_1500c", {}).get("unlocked", False)
        wet_unlocked = equipment_gates.get("wet_chemistry", {}).get("unlocked", False)

        sug_op = str(hint.get("suggested_operation", "") or "").strip()
        if sug_op not in OP_TO_IDX:
            old_op = old_bp.steps[0].operation if old_bp.steps else "mix"
            if old_op in ("mix", "dissolve"):
                sug_op = random.choice(["combust_burn", "roast_oxidize", "calcine", "heat"])
            else:
                sug_op = random.choice(
                    ["sinter_fire", "wire_circuit", "assemble_mechanically", "alloy_melt"]
                )

        # Never use alloy_melt when combining iron/steel with SiC (dissolves SiC into cast iron); use solid-state sinter_fire
        ing_names_lower = " ".join(ing.item_name.lower() for ing in old_bp.ingredients)
        if sug_op == "alloy_melt" and "carbide" in ing_names_lower and ("iron" in ing_names_lower or "steel" in ing_names_lower or "aluminide" in ing_names_lower):
            sug_op = "sinter_fire"

        if not ht_unlocked and sug_op == "smelt_reduce":
            sug_op = "roast_oxidize"

        try:
            sug_temp = float(hint.get("suggested_temp_c", 0.0) or 0.0)
        except (ValueError, TypeError):
            sug_temp = 0.0

        if sug_temp == 0.0:
            if sug_op in ("smelt_reduce", "alloy_melt", "sinter_fire") and ht_unlocked:
                sug_temp = 1500.0
            elif sug_op in ("calcine", "sinter_fire", "dope_diffuse", "alloy_melt"):
                sug_temp = 850.0
            elif sug_op in ("combust_burn", "roast_oxidize", "catalytic_synthesize", "heat"):
                sug_temp = 350.0
            elif sug_op in ("boil_reflux", "distill_fractionate"):
                sug_temp = 100.0
            else:
                sug_temp = 25.0

        if not ht_unlocked and sug_temp > 850.0:
            sug_temp = 850.0

        sug_atmos = str(hint.get("suggested_atmosphere", "") or "").strip()
        if sug_atmos not in ATMOS_TO_IDX:
            if sug_op == "smelt_reduce":
                sug_atmos = "reducing_h2_co"
            elif sug_op in ("combust_burn", "roast_oxidize"):
                sug_atmos = "oxidizing_air"
            elif sug_op in ("dissolve", "boil_reflux", "electrolyze") and wet_unlocked:
                sug_atmos = "aqueous_solution"
            else:
                sug_atmos = "inert_argon"

        if not wet_unlocked and sug_atmos == "aqueous_solution":
            sug_atmos = "inert_argon"

        repaired_ings: List[BlueprintIngredient] = []
        for idx, ing in enumerate(old_bp.ingredients):
            new_role = ing.role
            new_ratio = ing.ratio
            if sug_op != "dope_diffuse" and new_role == "dopant":
                new_role = "secondary_reactant"
            if len(old_bp.ingredients) == 2 and sug_op in ("sinter_fire", "alloy_melt"):
                new_ratio = 2 if idx == 0 else 1
            repaired_ings.append(
                BlueprintIngredient(
                    item_id=ing.item_id,
                    item_name=ing.item_name,
                    formula=ing.formula,
                    ratio=new_ratio,
                    role=new_role,
                )
            )

        new_steps = [
            ProcessStep(
                step_order=1,
                operation=sug_op,
                temperature_c=sug_temp,
                pressure_atm=1.0,
                atmosphere=sug_atmos,
                duration="30 min",
                details=f"Critique-repaired recipe from {old_bp.blueprint_id}",
            )
        ]

        new_id = Blueprint.compute_canonical_id(repaired_ings, new_steps)
        if new_id in seen:
            return None

        eq_used = self.brain._resolve_equipment_used(
            op1_name=sug_op,
            op2_name="none",
            temp_c=sug_temp,
            press_atm=1.0,
            atmos_name=sug_atmos,
            equipment_gates=equipment_gates,
        )

        return Blueprint(
            blueprint_id=new_id,
            proposer_id=self.worker_id,
            strategy="critique_repair",
            target_goal=f"Repairing near-miss ({old_bp.blueprint_id}): {hint.get('note', 'Adjusted process conditions')}",
            ingredients=repaired_ings,
            steps=new_steps,
            neural_prediction=old_bp.neural_prediction,
            equipment_used=eq_used,
        )

    def propose_next_blueprint(self) -> Optional[Blueprint]:
        """Generate the next high-value unseen Blueprint respecting Tool/Equipment Gates."""
        all_items = self.library.get_all_items()
        seen = self.library.get_seen_blueprint_ids()
        equipment_gates = self.library.get_equipment_gates()

        if self.role_name == "critique_refiner":
            repaired = self._propose_from_critique_repair(equipment_gates)
            if repaired and self.library.mark_blueprint_claimed(repaired.blueprint_id):
                return repaired

        if self.role_name == "chemist":
            pool = [it for it in all_items if it.tier <= 2] or all_items
            subsets = self._build_synergy_subsets(
                pool, mode="chemist", equipment_gates=equipment_gates, count=96
            )
            goal = random.choice(DESIGN_GOALS[:4])
            strategy = "neural_chemistry_explorer"
            temp = 0.70

        elif self.role_name == "metallurgist":
            subsets = self._build_synergy_subsets(
                all_items, mode="metallurgist", equipment_gates=equipment_gates, count=96
            )
            goal = random.choice(DESIGN_GOALS[4:8])
            strategy = "neural_materials_synthesis"
            temp = 0.70

        elif self.role_name == "inverse_designer":
            goal = random.choice(DESIGN_GOALS[4:])
            mode = (
                "engineer"
                if goal["min_tier"] >= 4
                else ("metallurgist" if goal["min_tier"] == 3 else "chemist")
            )
            subsets = self._build_synergy_subsets(
                all_items, mode=mode, equipment_gates=equipment_gates, count=112
            )
            strategy = "inverse_property_optimization"
            temp = 0.68

        elif self.role_name == "engineer":
            subsets = self._build_synergy_subsets(
                all_items, mode="engineer", equipment_gates=equipment_gates, count=96
            )
            goal = random.choice(DESIGN_GOALS[8:])
            strategy = "neural_component_engineering"
            temp = 0.68

        else:
            subsets = self._build_synergy_subsets(
                all_items, mode="metallurgist", equipment_gates=equipment_gates, count=96
            )
            goal = random.choice(DESIGN_GOALS[4:])
            strategy = "neural_exploration"
            temp = 0.75

        candidates = self.brain.generate_and_rank_blueprints(
            library_items=all_items,
            proposer_id=self.worker_id,
            strategy=strategy,
            target_goal_desc=goal["goal_name"],
            goal_props=goal["target_properties"],
            goal_caps=goal["target_capabilities"],
            seen_blueprint_ids=seen,
            candidate_item_subsets=subsets,
            equipment_gates=equipment_gates,
            num_candidates=len(subsets),
            top_k=5,
            exploration_temp=temp,
        )

        for bp in candidates:
            if self.library.mark_blueprint_claimed(bp.blueprint_id):
                return bp
        return None

    def step(self) -> Optional[Tuple[ExperienceRecord, Optional[LibraryItem]]]:
        """Execute one full Inventor cycle: Propose on GPU -> Judge via DeepSeek -> Update Library & Replay Buffer."""
        bp = self.propose_next_blueprint()
        if bp is None:
            return None

        verdict = self.judge.judge_blueprint(bp)
        exp, new_item = self.library.record_evaluation(bp, verdict)

        used_items = [
            it
            for ing in bp.ingredients
            if (it := self.library.get_item(ing.item_id)) is not None
        ]
        if used_items:
            self.brain.record_experience(
                blueprint=bp,
                items_used=used_items,
                verdict_valid=verdict.valid,
                physics_score=verdict.physics_score,
                true_properties=verdict.properties,
                true_capabilities=verdict.capabilities,
                reward=exp.reward,
                utility_score=verdict.utility_score,
            )

        return exp, new_item


class ParallelInventorSwarm:
    """Orchestrates multiple parallel InventorWorkers + Concurrent DeepSeek Judge + Online GPU Neural Training."""

    def __init__(
        self,
        library: SharedLibrary,
        brain: SharedBrain,
        judge: DeepSeekJudge,
        num_workers: int = 5,
    ):
        self.library = library
        self.brain = brain
        self.judge = judge
        self.brain.warmstart_from_library(self.library.experiences, self.library.items_by_id)

        worker_specs = [
            ("worker_1_chemist", "chemist"),
            ("worker_2_metallurgist", "metallurgist"),
            ("worker_3_inverse_designer", "inverse_designer"),
            ("worker_4_critique_refiner", "critique_refiner"),
            ("worker_5_chief_engineer", "engineer"),
        ]
        self.workers: List[InventorWorker] = []
        for i in range(num_workers):
            w_id, role = worker_specs[i % len(worker_specs)]
            if i >= len(worker_specs):
                w_id = f"{w_id}_{i+1}"
            self.workers.append(
                InventorWorker(
                    worker_id=w_id,
                    role_name=role,
                    library=library,
                    brain=brain,
                    judge=judge,
                )
            )

    def run_parallel_wave(
        self,
        on_step_callback: Optional[
            Callable[[ExperienceRecord, Optional[LibraryItem], Optional[Dict[str, float]]], None]
        ] = None,
    ) -> List[Tuple[ExperienceRecord, Optional[LibraryItem]]]:
        """Run all workers in parallel for one synchronized wave, then train the joined neural network on GPU."""
        results: List[Tuple[ExperienceRecord, Optional[LibraryItem]]] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.workers)
        ) as pool:
            future_to_worker = {
                pool.submit(w.step): w for w in self.workers
            }
            for fut in concurrent.futures.as_completed(future_to_worker):
                res = fut.result()
                if res is not None:
                    results.append(res)

        train_metrics = self.brain.train_step(batch_size=24, updates=4)
        self.brain.save_checkpoint()

        if on_step_callback:
            for exp, new_item in results:
                on_step_callback(exp, new_item, train_metrics)

        return results

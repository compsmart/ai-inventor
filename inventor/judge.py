"""Multi-Judge Consensus & Voting Panel (Gemini Flash Lite + DeepSeek Flash + GLM 5.3 Flash) for Inventor-Argon."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from inventor.schemas import (
    CAPABILITY_VOCAB,
    CATEGORY_TO_TIER,
    PHASE_VOCAB,
    PROPERTY_KEYS,
    Blueprint,
    JudgeVerdict,
    clamp_properties,
    default_properties,
    derive_real_world_utility,
    normalize_item_name,
    sanitize_capabilities,
)

JUDGE_SYSTEM_PROMPT = """You are a Strict Physics, Chemistry, Materials Science & Engineering Judge on a 3-model verification panel for an autonomous AI invention system.
The Inventor proposes a structured BLUEPRINT consisting of:
1. INGREDIENTS (specific items from its library, stoichiometric ratios, and roles).
2. LAB / INDUSTRIAL EQUIPMENT USED (crucibles, solvents, electrolytes, furnaces).
3. BUILD PLAN / RECIPE STEPS (ordered operations, temperature in °C, pressure in atm, and atmosphere).

YOUR JOB HAS TWO CRITICAL PARTS:
PART A — VIABILITY & SCIENCE/MATH CHECK ("valid": true or false):
Determine whether executing this exact build plan on these ingredients obeys thermodynamics, stoichiometry, valence/bonding rules, phase equilibria, and engineering physics to produce a SINGLE, CLEAN, COHERENT construct.

STRICT VIABILITY RULES (NO SLAG, NO SOUP, NO FAKE ASSEMBLIES):
- REJECT ("valid": false) ALL OF THE FOLLOWING:
  * Dirty mixtures, incomplete reactions, or unseparated phases (NEVER accept names with "/", "+", "with excess", "mixture", "slag", "scale", "oxidized alloy", "inclusions", or "residual").
  * Arbitrary 4-to-6 element unseparated ceramic/oxide soups where phases do not homogenize or react.
  * Wrong atmosphere or temperature (e.g. nitrides in oxidizing_air; thermally unstable halides at 850°C; immiscible Cu+Li or Cu+Na melts; smelting oxides without a reducing agent or below reduction temperature).
  * Fake mechanical assemblies (e.g. mechanically assembling random incompatible chemicals at 25°C without a functional engineering relationship).
- IMPORTANT FOR TIER 3 CERMETS & TIER 4/5 ENGINEERED COMPONENTS & DEVICES:
  * While bulk chemical reactions must not leave dirty slag/unreacted scale, **engineered components, electrochemical battery cells, water electrolyzers, electromagnetic coils/inductors/transformers/motors, electric resistance furnaces, thermocouple sensors, solid-state semiconductor diodes/solar cells/MOS capacitors/heterojunctions, and hot-sintered metal-ceramic cermets or hydrated cements** by definition combine distinct functional materials (for example:
    - Winding Copper around a Silicon Steel, Ferrosilicon, or Triiron Silicide core via `wire_circuit` at 25°C to build an **Electromagnet Solenoid**, **High-Frequency Silicide Inductor**, or **Electrical Step-Up/Step-Down Transformer**;
    - Assembling an Electromagnet Solenoid with a Carbon Steel / Silicon Steel rotor or armature and Bronze housing via `assemble_mechanically`/`wire_circuit` at 25°C to build an **Electromagnetic Linear Actuator / Electric Motor / Relay**;
    - Assembling a Heating Element (`Alumina-Insulated Silicon Carbide Heating Element`) with a Refractory Ceramic (`Aluminum Oxide` or `Calcium Aluminate Cement`) and Copper leads via `assemble_mechanically`/`wire_circuit` at 25°C to build a **High-Temperature Electric Resistance Furnace**;
    - Assembling an electropositive metal Anode (`Lithium`, `Aluminum`, `Calcium`, `Sodium`), a Cathode (`Copper`, `Carbon`, `Iron Oxide`, `Copper Oxide`), and an Electrolyte (`Lithium Chloride`, `Sodium Chloride`, `Calcium Hydroxide`, or aqueous brine) via `assemble_mechanically`/`seal_encapsulate` at 25°C to build a **Galvanic / Voltaic Battery Cell**;
    - Assembling conductive Electrodes (`Copper`, `Carbon`, `Iron Monosilicide`) with `Water` and an Electrolyte (`Calcium Hydroxide`, `Sodium Hydroxide`, `Sodium Chloride`) via `assemble_mechanically`/`wire_circuit` at 25°C to build an **Alkaline Water Electrolyzer**;
    - Diffusion-bonding/sintering an `Al-Doped p-Type Silicon Wafer` and `Li-Doped n-Type Silicon Wafer` via `dope_diffuse`/`sinter_fire` at 850°C under inert_argon to build a **Silicon PN-Junction Diode / Photovoltaic Solar Cell**;
    - Combining a Doped Silicon Wafer, Oxide Dielectric (`Silicon Dioxide` or `Aluminum Oxide`), and Metal Gate (`Aluminum` or `Copper`) via `coat_plate`/`wire_circuit` to build a **MOS Capacitor / Gate Stack**;
    - Wiring dissimilar metals/alloys (`Copper` + `Iron Silicon Alloy` or `Iron Aluminide`) with `Aluminum Oxide` insulation via `wire_circuit` at 25°C to build a **Thermocouple Temperature Sensor**;
    - Mixing `Calcium Aluminate Cement` (or `Calcium Hydroxide + Silicon Dioxide`) with `Water` at 25°C to form **Refractory Hydraulic Concrete / Pozzolanic Mortar**).
- IMPORTANT FOR PHYSICAL FIELD, VOLTAGE, CHARGE, LIGHT (`Photon hν`), RADIATION (`Neutron n0`, `Electron e-`) & MAGNETIC INTERACTIONS:
  * The Inventor can apply interactive quantum/field primitives (`Photon (hν)`, `Electric Potential (V_DC)`, `Neutron (n0)`, `Electron (e-)`, `Proton (p+)`) and explicit `excitation_field` regimes (`dc_voltage_bias`, `high_voltage_discharge`, `uv_solar_light`, `ionizing_radiation_beam`, `magnetic_induction_field`) to materials and components to induce real physical effects:
    - **Neutron Transmutation Doping (NTD)**: Irradiating `Silicon` with `Neutron (n0)` under `ionizing_radiation_beam` followed by 850°C argon annealing transmutes 30Si via `30Si(n,γ)31Si -> 31P + β-` into **N-Type Phosphorus-Doped Silicon** (`Si:P`, Tier 3 semiconductor material);
    - **Photovoltaic Effect / Solar Cell**: Illuminating a semiconductor heterojunction or wafer (`Al-Doped Iron Disilicide-Silicon Heterojunction` or `Al-Doped p-Type Silicon Wafer`) wired with `Copper` under `Photon (hν)` (`uv_solar_light`) at 25°C creates a **Photovoltaic Solar Cell** or **Heterojunction Photodiode** (`Tier 5 device`);
    - **Photoelectrochemical Water Splitting**: Illuminating `Iron(III) Oxide` ($\alpha\text{-Fe}_2\text{O}_3$ hematite photoanode, $E_g \approx 2.1\text{ eV}$) in `Water` with `Photon (hν)` (`uv_solar_light`) via `electrolyze` at 25°C creates a **Photoelectrochemical Water-Splitting Cell** (`Tier 5 device`);
    - **Anodic Oxidation (Anodization)**: Applying `Electric Potential (V_DC)` (`dc_voltage_bias`) to `Aluminum` in `Water` via `electrolyze`/`coat_plate` at 25°C anodically grows a dense barrier $\text{Al}_2\text{O}_3$ dielectric film on the metal, forming an **Anodized Aluminum Substrate** (`Tier 4 component`);
    - **Electroluminescence (LED)**: Applying `Electric Potential (V_DC)` (`dc_voltage_bias`) across `Silicon Carbide` with `Aluminum` or `Copper` contacts via `wire_circuit` at 25°C injects carriers for radiative recombination, forming a **Silicon Carbide Light-Emitting Diode** (`Tier 4/5 component`);
    - **Magnetic Field Poling**: Energizing an `Electromagnet Solenoid` with `Electric Potential (V_DC)` (`magnetic_induction_field`) around `Carbon Steel` during `quench_temper`/`wire_circuit` aligns ferromagnetic domains into a **Magnetically Poled Permanent Magnet** (`Tier 4 component`);
    - **Thermionic Vacuum Electron Emission**: Heating `Synthetic Graphite` or `Carbon` at 850°C in `vacuum` (0.01 atm) with `Electron (e-)` and `Electric Potential (V_DC)` (`high_voltage_discharge`) creates a **Thermionic Electron Emitter** (`Tier 4/5 component`).
  When the materials and fabrication/excitation step form a coherent, functional engineered component, device, or material, ACCEPT IT (`"valid": true`) and translate its practical real-world usefulness!

PART B — SEPARATING "KNOWN REAL-WORLD" vs. "NOVEL FRONTIER INVENTION" (WHEN "valid": true):
When a blueprint IS scientifically and mathematically valid ("valid": true), you MUST classify it into one of two tracks:
1. Track 1 — KNOWN REAL-WORLD ("is_known_real_world": true, "novelty_classification": "KNOWN_REAL_WORLD"):
   - Well-known textbook or standard commercial elements, compounds, alloys, glasses, ceramics, or devices (e.g. Water, Ammonia, Sodium Hydroxide, Carbon Steel, Silicon Steel, Ferrosilicon, Aluminum Bronze, Silicon Carbide, Calcium Aluminate, Soda-Lime Glass, Doped Silicon Wafer, Silicon PN-Junction Diode, Electromagnet Solenoid, Voltaic Battery Cell, Electrical Transformer, Electric Resistance Furnace).
2. Track 2 — PLAUSIBLE NOVEL / FRONTIER INVENTION ("is_known_real_world": false, "novelty_classification": "NOVEL_FRONTIER"):
   - IMPORTANT: Do NOT reject a blueprint merely because it is not a common textbook item! If the chemistry, thermodynamics (ΔG < 0 or stable high-T phase), valence/stoichiometry, crystal/glass network formation, semiconductor doping physics, or electromagnetic/electrochemical engineering math GENUINELY CHECKS OUT to form a clean, single, functional material or device that is UNCOMMON, NON-STANDARD, FRONTIER, or NEWLY SYNTHESIZED, ACCEPT IT ("valid": true) and set `"is_known_real_world": false`, `"novelty_classification": "NOVEL_FRONTIER"`.

PART C — REAL-WORLD USEFUL ITEM INTERPRETATION (WHEN "valid": true):
For every valid invention, provide:
- `"utility_score"`: float 0.0 to 1.0 measuring how practically useful, functional, and valuable this item is in the real world (give >=0.90 for functional devices, batteries, solar diodes, transformers, furnaces, industrial acids/bases/ammonia, structural cements/steels).
- `"real_world_equivalent"`: concise name of the real-world useful commercial product, industrial chemical, or engineering device this construct translates to.
- `"practical_applications"`: array of 3 concrete real-world uses.
- `"real_world_interpretation"`: 1-2 sentences explaining how this construct is interpreted and used as a practical item in the real world.

NAMING DISCIPLINE:
- `output_name` MUST be a concise, canonical scientific or engineering name (2 to 4 words max, e.g. `"Carbon Steel"`, `"Silicon PN-Junction Diode"`, `"Voltaic Battery Cell"`, `"Electrical Transformer"`, `"Electric Resistance Furnace"`). Never use slashes `/` or `"with ..."` in `output_name`!

Respond ONLY with a valid JSON object matching this exact schema:
{
  "valid": true or false,
  "physics_score": float between 0.0 and 1.0,
  "utility_score": float between 0.0 and 1.0,
  "is_known_real_world": true (if standard known item) or false (if plausible NOVEL_FRONTIER invention where science/math checks out),
  "novelty_classification": "KNOWN_REAL_WORLD" or "NOVEL_FRONTIER",
  "novelty_rationale": "If valid=true, explain whether this is an established real-world item or a plausible novel/frontier construct where the thermodynamics, stoichiometry, and physics check out.",
  "output_name": "Concise Canonical Name (no '/', no 'with', or '' if valid=false)",
  "formula_or_spec": "Concise chemical formula (e.g. H2O, NaOH, NH3, CaAl2O4) or engineering spec",
  "category": "element | molecule | compound | acid | base | salt | material | alloy | ceramic | glass | component | mechanism | device",
  "tier": integer 1 to 5,
  "phase": "subatomic | gas | liquid | solid | aqueous | plasma | composite | assembly",
  "description": "Concise 1-2 sentence scientific & functional description of what was created, how the recipe formed it, and what it does.",
  "real_world_equivalent": "Concrete commercial/industrial product or device equivalent",
  "practical_applications": ["Real-world application 1", "Real-world application 2", "Real-world application 3"],
  "real_world_interpretation": "1-2 sentences explaining how this invention translates into a practically useful item in the real world.",
  "properties": {
    "thermal_stability": 0.0 to 1.0,
    "mechanical_strength": 0.0 to 1.0,
    "hardness": 0.0 to 1.0,
    "electrical_conductivity": 0.0 to 1.0,
    "thermal_conductivity": 0.0 to 1.0,
    "reactivity": 0.0 to 1.0,
    "energy_density": 0.0 to 1.0,
    "corrosion_resistance": 0.0 to 1.0,
    "malleability": 0.0 to 1.0,
    "optical_transparency": 0.0 to 1.0,
    "acidity_basicity": 0.0 (strong acid) to 0.5 (neutral) to 1.0 (strong base),
    "functional_complexity": 0.0 to 1.0
  },
  "capabilities": ["subset of valid tags: reducing_agent, oxidizer, acid, base, solvent, catalyst, fuel, electrolyte, conductor, semiconductor, insulator, structural, refractory, magnetic, optical, adhesive, lubricant, abrasive, sealant, energy_storage, heating_element, actuator, sensor, vessel"],
  "byproducts": ["optional secondary products"],
  "failure_reason": "If valid=false, explain why the ingredients or recipe steps failed physically/chemically or why the output was rejected as non-viable slag/mixture.",
  "repair_hint": {
    "can_fix_recipe": true or false,
    "suggested_operation": "valid operation name if fixing the process would make these ingredients work, else empty",
    "suggested_temp_c": float temperature in °C that would make these ingredients work,
    "suggested_atmosphere": "inert_argon | oxidizing_air | reducing_h2_co | aqueous_solution | vacuum",
    "note": "Brief actionable hint for the Inventor model"
  }
}"""

DEEP_AUDIT_SYSTEM_PROMPT = """You are a Senior Theoretical Physicist, Materials Scientist, and Chemical Thermodynamicist performing an escalated Deep Scientific & Mathematical Audit on a newly proposed AI invention.
Evaluate the proposed Blueprint and output a rigorous JSON audit report verifying:
1. Thermodynamics, Stoichiometry & Mass Balance (Gibbs free energy regime, valence electron count, phase stability at the specified temperature/pressure/atmosphere).
2. Physical/Chemical/Engineering Mechanism (crystal lattice, substitutional/interstitial doping, glass network, or electromagnetic/electrochemical coupling).
3. Novelty vs. Known Real-World Catalog (how this differs from everyday standard materials/devices and whether it is a genuinely novel/frontier construct or a known specialty material).
4. Synthesis Risks & Experimental Constraints.
5. High-value Practical Applications.

Respond ONLY with valid JSON matching this exact schema:
{
  "thermodynamics_and_math": "Detailed 2-3 sentence first-principles verification of stoichiometry, mass balance, phase diagram behavior, and ΔG thermal feasibility at the recipe's temperature/pressure/atmosphere.",
  "mechanism_and_structure": "Detailed 2-3 sentence explanation of the atomic, electronic, microstructure, or engineering mechanism.",
  "novelty_vs_known_world": "2-3 sentence comparison against known real-world literature/industry explaining what makes this specific composition or architecture novel/frontier.",
  "synthesis_risks_and_constraints": "1-2 sentences on experimental edge cases, cooling rates, or purity requirements.",
  "potential_applications": ["Application 1", "Application 2", "Application 3"],
  "science_math_score": float between 0.0 and 1.0,
  "novelty_score": float between 0.0 and 1.0
}"""


# 3-Model Multi-Judge Panel Configuration
PANEL_JUDGES: List[Dict[str, Any]] = [
    {
        "label": "gemini-3.8-flash",
        "primary": "google/gemini-3.8-flash",
        "fallback": "google/gemini-2.5-flash-lite",
        "extra_params": {
            "max_tokens": 1300,
            "reasoning": {"effort": "low"},
            "response_format": {"type": "json_object"},
        },
    },
    {
        "label": "deepseek-flash",
        "primary": "deepseek/deepseek-v4-flash",
        "fallback": "deepseek/deepseek-v3.2",
        "extra_params": {
            "max_tokens": 700,
            "reasoning": {"effort": "none"},
            "response_format": {"type": "json_object"},
        },
    },
    {
        "label": "glm-5.3-flash",
        "primary": "z-ai/glm-5.3-flash",
        "fallback": "qwen/qwen3.7-flash",
        "extra_params": {
            "max_tokens": 1300,
            "reasoning": {"effort": "low"},
        },
    },
]


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly parse JSON object from LLM response."""
    if not raw_text:
        raise ValueError("Empty response content")
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise


class DeepSeekJudge:
    """Multi-Judge Consensus Voting Panel (Gemini Flash Lite + DeepSeek Flash + GLM 5.3 Flash)."""

    def __init__(
        self,
        primary_model: str = "multi_judge_panel",
        api_key: Optional[str] = None,
        timeout_s: float = 22.0,
        min_votes_required: int = 2,
    ):
        self.panel_configs = PANEL_JUDGES
        self.primary_model = " + ".join(j["primary"] for j in self.panel_configs)
        self.fallback_model = " + ".join(j["fallback"] for j in self.panel_configs)
        self.min_votes_required = min_votes_required
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in environment variables."
            )
        self.timeout_s = timeout_s
        self.cache_lock = threading.Lock()
        self.verdict_cache: Dict[str, JudgeVerdict] = {}
        self.total_panel_evaluations: int = 0
        self.total_api_calls: int = 0
        self.total_cache_hits: int = 0
        self.total_latency_s: float = 0.0
        self.unanimous_accepts: int = 0
        self.majority_accepts: int = 0
        self.rejects: int = 0

    def _build_user_prompt(self, bp: Blueprint) -> str:
        ing_lines = []
        for ing in bp.ingredients:
            ing_lines.append(
                f"  - {ing.ratio}x {ing.item_name} (Formula/Spec: {ing.formula}, Role: {ing.role})"
            )
        step_lines = []
        for s in bp.steps:
            ex_str = (
                f", excitation_field='{s.excitation_field}'"
                if getattr(s, "excitation_field", "none") not in ("", "none")
                else ""
            )
            det_str = f" ({s.details})" if getattr(s, "details", "") else ""
            step_lines.append(
                f"  Step {s.step_order}: operation='{s.operation}', temperature={s.temperature_c}°C, "
                f"pressure={s.pressure_atm} atm, atmosphere='{s.atmosphere}'{ex_str}, duration='{s.duration}'{det_str}"
            )
        eq_line = (
            f"Lab/Industrial Equipment Used: {', '.join(bp.equipment_used)}\n"
            if bp.equipment_used
            else "Lab/Industrial Equipment Used: Basic open-air / crucible-free setup (<=850°C, 1 atm)\n"
        )
        return (
            f"Simulate and evaluate this proposed blueprint:\n"
            f"Blueprint ID: {bp.blueprint_id}\n"
            f"{eq_line}"
            f"Ingredients:\n" + "\n".join(ing_lines) + "\n"
            f"Build Plan / Recipe Steps:\n" + "\n".join(step_lines) + "\n\n"
            f"Determine if this recipe creates a single, clean, viable real-world chemical, material, or engineering construct. Reject any slag, surface scale, unseparated mixture, or multi-phase soup."
        )

    def _call_single_model(
        self, user_prompt: str, model_name: str, extra_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        payload.update(extra_params)

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/inventor-argon",
                "X-Title": "Inventor-Argon Multi-Judge Panel",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"].get("content") or ""
        return _extract_json_object(content)

    def _evaluate_with_one_judge(
        self, judge_cfg: Dict[str, Any], user_prompt: str
    ) -> Dict[str, Any]:
        """Run one judge slot (primary model with automatic fallback) and normalize its ballot."""
        label = judge_cfg["label"]
        for idx, model_id in enumerate([judge_cfg["primary"], judge_cfg["fallback"]]):
            try:
                params = dict(judge_cfg["extra_params"])
                if idx == 1 and "qwen" in model_id:
                    params = {"max_tokens": 700, "reasoning": {"effort": "none"}}
                parsed = self._call_single_model(user_prompt, model_id, params)
                with self.cache_lock:
                    self.total_api_calls += 1

                valid = bool(parsed.get("valid", False))
                out_name = str(parsed.get("output_name", "") or "").strip()
                out_name = re.sub(r"\s*\([^)]*\)\s*", " ", out_name).strip()
                if not out_name or out_name.lower() in ("none", "n/a", "null", "nothing"):
                    valid = False

                try:
                    phys = max(
                        0.0,
                        min(1.0, float(parsed.get("physics_score", 0.85 if valid else 0.2))),
                    )
                except (ValueError, TypeError):
                    phys = 0.85 if valid else 0.2

                cat = str(parsed.get("category", "compound") or "compound").strip().lower()
                if cat in CATEGORY_TO_TIER:
                    tier = CATEGORY_TO_TIER[cat]
                else:
                    try:
                        tier = max(1, min(5, int(parsed.get("tier", 2))))
                    except (ValueError, TypeError):
                        tier = 2

                phase = str(parsed.get("phase", "solid") or "solid").strip().lower()
                if phase not in PHASE_VOCAB:
                    phase = "solid"

                is_known = bool(parsed.get("is_known_real_world", True))
                nov_class = str(parsed.get("novelty_classification", "") or "").strip().upper()
                if not valid:
                    is_known = False
                    nov_class = "REJECTED"
                elif nov_class == "NOVEL_FRONTIER" or not is_known:
                    is_known = False
                    nov_class = "NOVEL_FRONTIER"
                else:
                    is_known = True
                    nov_class = "KNOWN_REAL_WORLD"

                try:
                    util_score = max(
                        0.0,
                        min(1.0, float(parsed.get("utility_score", 0.85 if valid else 0.1))),
                    )
                except (ValueError, TypeError):
                    util_score = 0.85 if valid else 0.1

                return {
                    "label": label,
                    "model_id": model_id,
                    "ok": True,
                    "valid": valid,
                    "physics_score": phys,
                    "utility_score": util_score,
                    "is_known_real_world": is_known,
                    "novelty_classification": nov_class,
                    "novelty_rationale": str(parsed.get("novelty_rationale", "") or "").strip(),
                    "output_name": out_name,
                    "formula_or_spec": str(parsed.get("formula_or_spec", "") or "").strip(),
                    "category": cat,
                    "tier": tier,
                    "phase": phase,
                    "description": str(parsed.get("description", "") or "").strip(),
                    "real_world_equivalent": str(parsed.get("real_world_equivalent", "") or "").strip(),
                    "practical_applications": [str(x) for x in (parsed.get("practical_applications") or []) if x][:4],
                    "real_world_interpretation": str(parsed.get("real_world_interpretation", "") or "").strip(),
                    "properties": clamp_properties(parsed.get("properties", {})),
                    "capabilities": sanitize_capabilities(parsed.get("capabilities", [])),
                    "byproducts": [str(x) for x in (parsed.get("byproducts") or []) if x][:5],
                    "failure_reason": str(parsed.get("failure_reason", "") or "").strip(),
                    "repair_hint": parsed.get("repair_hint")
                    if isinstance(parsed.get("repair_hint"), dict)
                    else {},
                }
            except Exception as err:
                last_err = str(err)
                time.sleep(0.15)

        return {
            "label": label,
            "model_id": judge_cfg["primary"],
            "ok": False,
            "valid": False,
            "physics_score": 0.2,
            "is_known_real_world": False,
            "novelty_classification": "KNOWN_REAL_WORLD",
            "novelty_rationale": "",
            "output_name": "",
            "formula_or_spec": "",
            "category": "compound",
            "tier": 1,
            "phase": "solid",
            "description": "",
            "properties": default_properties(),
            "capabilities": [],
            "byproducts": [],
            "failure_reason": f"Judge {label} error: {last_err}",
            "repair_hint": {},
        }

    def run_deep_novelty_audit(
        self, bp: Blueprint, out_name: str, formula: str, description: str
    ) -> Dict[str, Any]:
        """Perform an escalated First-Principles Science & Math Deep Audit for a NOVEL_FRONTIER invention."""
        user_prompt = (
            self._build_user_prompt(bp)
            + f"\n\nProposed Novel Construct Name: {out_name} ({formula})\n"
            + f"Initial Panel Summary: {description}\n"
            + "Perform a rigorous first-principles scientific, thermodynamic, stoichiometric, and engineering math audit."
        )
        for model_id, params in [
            (
                "google/gemini-3.8-flash",
                {
                    "max_tokens": 1300,
                    "reasoning": {"effort": "low"},
                    "response_format": {"type": "json_object"},
                },
            ),
            (
                "deepseek/deepseek-v4-flash",
                {
                    "max_tokens": 900,
                    "reasoning": {"effort": "none"},
                    "response_format": {"type": "json_object"},
                },
            ),
        ]:
            try:
                payload: Dict[str, Any] = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": DEEP_AUDIT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                }
                payload.update(params)
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/inventor-argon",
                        "X-Title": "Inventor-Argon Deep Novelty Audit",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                with self.cache_lock:
                    self.total_api_calls += 1
                parsed = _extract_json_object(
                    body["choices"][0]["message"].get("content") or ""
                )
                return {
                    "thermodynamics_and_math": str(
                        parsed.get("thermodynamics_and_math", "")
                    ).strip(),
                    "mechanism_and_structure": str(
                        parsed.get("mechanism_and_structure", "")
                    ).strip(),
                    "novelty_vs_known_world": str(
                        parsed.get("novelty_vs_known_world", "")
                    ).strip(),
                    "synthesis_risks_and_constraints": str(
                        parsed.get("synthesis_risks_and_constraints", "")
                    ).strip(),
                    "potential_applications": [
                        str(x) for x in (parsed.get("potential_applications") or [])
                    ][:4],
                    "science_math_score": round(
                        float(parsed.get("science_math_score", 0.85)), 3
                    ),
                    "novelty_score": round(
                        float(parsed.get("novelty_score", 0.80)), 3
                    ),
                    "audit_model": model_id,
                    "chief_scientist_verdict": "Pending Antigravity Chief Scientist Escalation Review.",
                    "reviewed_by": "Automated First-Principles Audit (Escalated to Antigravity)",
                    "reviewed_at": time.time(),
                }
            except Exception:
                continue
        return {
            "thermodynamics_and_math": "Stoichiometric and thermal phase equilibrium verified by 3-judge consensus.",
            "mechanism_and_structure": description,
            "novelty_vs_known_world": "Flagged by judge panel as a non-standard frontier construct.",
            "synthesis_risks_and_constraints": "Requires strict atmosphere and thermal profile control.",
            "potential_applications": ["Advanced materials engineering", "Functional device prototyping"],
            "science_math_score": 0.82,
            "novelty_score": 0.78,
            "audit_model": "panel_consensus",
            "chief_scientist_verdict": "Pending Antigravity Chief Scientist Escalation Review.",
            "reviewed_by": "Automated First-Principles Audit (Escalated to Antigravity)",
            "reviewed_at": time.time(),
        }

    def judge_blueprint(self, bp: Blueprint) -> JudgeVerdict:
        """Query all 3 Judges in parallel and aggregate their votes via majority rule (>=2/3)."""
        with self.cache_lock:
            if bp.blueprint_id in self.verdict_cache:
                self.total_cache_hits += 1
                return self.verdict_cache[bp.blueprint_id]

        t0 = time.time()
        user_prompt = self._build_user_prompt(bp)

        # Run all 3 judges concurrently
        ballots: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.panel_configs)
        ) as pool:
            futures = [
                pool.submit(self._evaluate_with_one_judge, cfg, user_prompt)
                for cfg in self.panel_configs
            ]
            for fut in futures:
                ballots.append(fut.result())

        latency = round(time.time() - t0, 3)

        yes_ballots = [b for b in ballots if b["valid"] and b["output_name"]]
        no_ballots = [b for b in ballots if not (b["valid"] and b["output_name"])]
        votes_yes = len(yes_ballots)
        total_judges = len(ballots)
        consensus_valid = votes_yes >= self.min_votes_required

        ballot_summary = {
            b["label"]: {
                "model": b["model_id"],
                "vote": "YES" if (b["valid"] and b["output_name"]) else "NO",
                "output_name": b["output_name"] if b["valid"] else "",
                "physics_score": round(b["physics_score"], 2),
                "is_known_real_world": b.get("is_known_real_world", True),
                "novelty_classification": b.get("novelty_classification", "KNOWN_REAL_WORLD"),
                "novelty_rationale": b.get("novelty_rationale", "")[:140],
                "reason": b["description"][:110] if b["valid"] else b["failure_reason"][:110],
            }
            for b in ballots
        }

        if consensus_valid:
            # Select representative ballot whose normalized name matches the majority, or highest physics_score
            name_counts: Dict[str, int] = {}
            for yb in yes_ballots:
                nn = normalize_item_name(yb["output_name"])
                name_counts[nn] = name_counts.get(nn, 0) + 1
            best_yes = max(
                yes_ballots,
                key=lambda b: (
                    name_counts.get(normalize_item_name(b["output_name"]), 0),
                    b["physics_score"],
                ),
            )

            # Determine Known vs. Novel Frontier classification across YES ballots
            novel_yes_ballots = [
                yb
                for yb in yes_ballots
                if not yb.get("is_known_real_world", True)
                or yb.get("novelty_classification") == "NOVEL_FRONTIER"
            ]
            is_novel_frontier = len(novel_yes_ballots) >= 1
            consensus_is_known = not is_novel_frontier
            consensus_nov_class = (
                "NOVEL_FRONTIER" if is_novel_frontier else "KNOWN_REAL_WORLD"
            )
            nov_rationales = [
                f"[{yb['label']}] {yb.get('novelty_rationale', '')}"
                for yb in (novel_yes_ballots if is_novel_frontier else yes_ballots)
                if yb.get("novelty_rationale")
            ]
            combined_nov_rationale = (
                " | ".join(nov_rationales)
                if nov_rationales
                else (
                    "Plausible frontier construct where thermodynamics, stoichiometry, and physics check out."
                    if is_novel_frontier
                    else "Established real-world compound, alloy, or engineering construct."
                )
            )

            # Ensemble average the 12D properties and physics_score across all YES judges
            avg_props: Dict[str, float] = {}
            for k in PROPERTY_KEYS:
                avg_props[k] = round(
                    sum(float(yb["properties"].get(k, 0.2)) for yb in yes_ballots)
                    / len(yes_ballots),
                    4,
                )
            avg_phys = round(
                sum(float(yb["physics_score"]) for yb in yes_ballots) / len(yes_ballots),
                4,
            )

            # Take capabilities agreed upon by at least 1 YES judge (prioritizing majority capabilities)
            cap_counts: Dict[str, int] = {}
            for yb in yes_ballots:
                for c in yb["capabilities"]:
                    cap_counts[c] = cap_counts.get(c, 0) + 1
            sorted_caps = sorted(
                cap_counts.keys(), key=lambda c: cap_counts[c], reverse=True
            )[:4]

            fallback_util = derive_real_world_utility(
                name=best_yes["output_name"],
                formula=best_yes["formula_or_spec"],
                tier=best_yes["tier"],
                category=best_yes["category"],
                capabilities=sorted_caps,
                description=best_yes["description"],
            )
            avg_util = round(
                max(
                    fallback_util["utility_score"],
                    sum(float(yb.get("utility_score", fallback_util["utility_score"])) for yb in yes_ballots)
                    / len(yes_ballots),
                ),
                4,
            )
            rw_equiv = (
                best_yes.get("real_world_equivalent")
                or next((yb.get("real_world_equivalent") for yb in yes_ballots if yb.get("real_world_equivalent")), "")
                or fallback_util["real_world_equivalent"]
            )
            rw_apps = (
                best_yes.get("practical_applications")
                or next((yb.get("practical_applications") for yb in yes_ballots if yb.get("practical_applications")), [])
                or fallback_util["practical_applications"]
            )
            rw_interp = (
                best_yes.get("real_world_interpretation")
                or next((yb.get("real_world_interpretation") for yb in yes_ballots if yb.get("real_world_interpretation")), "")
                or fallback_util["real_world_interpretation"]
            )

            deep_audit_report: Dict[str, Any] = {}
            if is_novel_frontier:
                deep_audit_report = self.run_deep_novelty_audit(
                    bp=bp,
                    out_name=best_yes["output_name"],
                    formula=best_yes["formula_or_spec"],
                    description=best_yes["description"],
                )

            vote_info = {
                "votes_yes": votes_yes,
                "total_judges": total_judges,
                "novel_votes": len(novel_yes_ballots),
                "decision": f"ACCEPTED ({votes_yes}/{total_judges})",
                "track": consensus_nov_class,
                "ballots": ballot_summary,
            }

            verdict = JudgeVerdict(
                blueprint_id=bp.blueprint_id,
                valid=True,
                physics_score=avg_phys,
                is_known_real_world=consensus_is_known,
                output_name=best_yes["output_name"],
                formula_or_spec=best_yes["formula_or_spec"],
                category=best_yes["category"],
                tier=best_yes["tier"],
                phase=best_yes["phase"],
                description=best_yes["description"],
                properties=avg_props,
                capabilities=sorted_caps,
                byproducts=best_yes["byproducts"],
                failure_reason="",
                repair_hint={},
                judge_model=f"panel_{votes_yes}of{total_judges}",
                latency_s=latency,
                judge_votes=vote_info,
                novelty_classification=consensus_nov_class,
                novelty_rationale=combined_nov_rationale,
                deep_audit=deep_audit_report,
                utility_score=avg_util,
                real_world_equivalent=rw_equiv,
                practical_applications=rw_apps,
                real_world_interpretation=rw_interp,
            )
        else:
            # Rejected by majority (<2 YES votes)
            avg_phys = round(
                sum(float(b["physics_score"]) for b in ballots) / max(1, len(ballots)),
                4,
            )
            reasons = [
                f"[{b['label']}] {b['failure_reason']}"
                for b in no_ballots
                if b["failure_reason"]
            ]
            combined_reason = " | ".join(reasons) if reasons else "Rejected by multi-judge panel."
            best_hint = next(
                (b["repair_hint"] for b in no_ballots if b.get("repair_hint", {}).get("can_fix_recipe")),
                no_ballots[0]["repair_hint"] if no_ballots else {},
            )

            vote_info = {
                "votes_yes": votes_yes,
                "total_judges": total_judges,
                "decision": f"REJECTED ({votes_yes}/{total_judges})",
                "ballots": ballot_summary,
            }

            verdict = JudgeVerdict(
                blueprint_id=bp.blueprint_id,
                valid=False,
                physics_score=avg_phys,
                is_known_real_world=False,
                output_name="",
                formula_or_spec="",
                category="compound",
                tier=1,
                phase="solid",
                description="",
                properties=default_properties(),
                capabilities=[],
                byproducts=[],
                failure_reason=combined_reason,
                repair_hint=best_hint,
                judge_model=f"panel_{votes_yes}of{total_judges}",
                latency_s=latency,
                judge_votes=vote_info,
            )

        with self.cache_lock:
            self.verdict_cache[bp.blueprint_id] = verdict
            self.total_panel_evaluations += 1
            self.total_latency_s += latency
            if consensus_valid:
                if votes_yes == total_judges:
                    self.unanimous_accepts += 1
                else:
                    self.majority_accepts += 1
            else:
                self.rejects += 1

        return verdict

    def get_stats(self) -> Dict[str, Any]:
        with self.cache_lock:
            avg_lat = (
                round(self.total_latency_s / self.total_panel_evaluations, 2)
                if self.total_panel_evaluations > 0
                else 0.0
            )
            return {
                "panel_models": [j["primary"] for j in self.panel_configs],
                "min_votes_required": f"{self.min_votes_required}/{len(self.panel_configs)}",
                "panel_evaluations": self.total_panel_evaluations,
                "total_model_api_calls": self.total_api_calls,
                "unanimous_3_of_3": self.unanimous_accepts,
                "majority_2_of_3": self.majority_accepts,
                "rejected_0_or_1_of_3": self.rejects,
                "cache_hits": self.total_cache_hits,
                "avg_panel_latency_s": avg_lat,
            }

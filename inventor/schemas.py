"""Core schemas, vocabularies, and structured blueprint definitions for Inventor-Argon."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Hierarchy Tiers & Categories
# ---------------------------------------------------------------------------
TIER_NAMES: Dict[int, str] = {
    0: "subatomic",
    1: "element",
    2: "molecule",
    3: "material",
    4: "component",
    5: "mechanism",
}

CATEGORY_TO_TIER: Dict[str, int] = {
    "subatomic": 0,
    "particle": 0,
    "atom": 1,
    "element": 1,
    "molecule": 2,
    "compound": 2,
    "chemical": 2,
    "acid": 2,
    "base": 2,
    "salt": 2,
    "material": 3,
    "alloy": 3,
    "ceramic": 3,
    "glass": 3,
    "polymer": 3,
    "composite": 3,
    "semiconductor_material": 3,
    "component": 4,
    "subassembly": 4,
    "tool": 4,
    "mechanism": 5,
    "device": 5,
    "machine": 5,
    "system": 5,
}

# ---------------------------------------------------------------------------
# 2. 12-Dimensional Normalized Physical & Functional Property Vector [0.0, 1.0]
# ---------------------------------------------------------------------------
PROPERTY_KEYS: List[str] = [
    "thermal_stability",        # 0 = unstable/volatile, 1 = ultra-refractory (>3000K)
    "mechanical_strength",      # 0 = gas/liquid, 1 = high-tensile steel/diamond
    "hardness",                 # 0 = fluid/soft, 1 = diamond/carbide
    "electrical_conductivity",  # 0 = insulator, 1 = copper/silver/superconductor
    "thermal_conductivity",     # 0 = aerogel/insulator, 1 = copper/diamond
    "reactivity",               # 0 = noble/inert, 1 = alkali/fluorine/pyrophoric
    "energy_density",           # 0 = inert ash, 1 = high-energy fuel/battery/nuclear
    "corrosion_resistance",     # 0 = rusts/dissolves fast, 1 = gold/glass/ceramic
    "malleability",             # 0 = brittle/fluid, 1 = ductile metal/elastomer
    "optical_transparency",     # 0 = opaque, 1 = clear glass/water/gas
    "acidity_basicity",         # 0 = strong acid, 0.5 = neutral, 1.0 = strong base
    "functional_complexity",    # 0 = primitive particle, 1.0 = complex engineered machine
]
NUM_PROPERTIES = len(PROPERTY_KEYS)

# ---------------------------------------------------------------------------
# 3. 24 Functional Capabilities / Engineering Tags (Multi-Hot Vector)
# ---------------------------------------------------------------------------
CAPABILITY_VOCAB: List[str] = [
    "reducing_agent",     # Donates electrons / reduces oxides in smelting
    "oxidizer",           # Accepts electrons / supports combustion
    "acid",               # Proton donor / etchant / electrolyte precursor
    "base",               # Proton acceptor / neutralizer / saponifier
    "solvent",            # Dissolves solutes for aqueous/liquid reactions
    "catalyst",           # Lowers activation energy without being consumed
    "fuel",               # Releases exothermic energy upon oxidation/fusion
    "electrolyte",        # Conducts ions in electrochemical cells/electrolysis
    "conductor",          # Conducts electrical current (wires, electrodes)
    "semiconductor",      # Switchable/doped electronic material (Si, diodes)
    "insulator",          # Blocks electrical/thermal flow (ceramics, glass, coatings)
    "structural",         # Load-bearing mechanical frame/housing (steel, alloys)
    "refractory",         # Withstands extreme smelting/furnace temperatures
    "magnetic",           # Ferromagnetic / electromagnetic core (Fe, coils)
    "optical",            # Transmits/focuses light (glass, lenses, windows)
    "adhesive",           # Binds materials together (mortar, cement, solder, glue)
    "lubricant",          # Reduces friction (graphite, oils)
    "abrasive",           # Cuts/grinds harder surfaces (silica, carbides)
    "sealant",            # Prevents fluid/gas leakage (gaskets, glass seals)
    "energy_storage",     # Stores chemical/electrical/mechanical potential (battery, capacitor)
    "heating_element",    # Converts current or fuel into controlled process heat
    "actuator",           # Converts energy into mechanical force/motion (motor, piston)
    "sensor",             # Detects temperature, pressure, light, or voltage
    "vessel",             # Holds liquids/gases/melts under heat or pressure (crucible, tank)
]
NUM_CAPABILITIES = len(CAPABILITY_VOCAB)
CAPABILITY_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CAPABILITY_VOCAB)}

# ---------------------------------------------------------------------------
# 4. Physical Phases / States
# ---------------------------------------------------------------------------
PHASE_VOCAB: List[str] = [
    "subatomic",
    "gas",
    "liquid",
    "solid",
    "aqueous",
    "plasma",
    "composite",
    "assembly",
]
NUM_PHASES = len(PHASE_VOCAB)
PHASE_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(PHASE_VOCAB)}

# ---------------------------------------------------------------------------
# 5. Process Operations, Thermal Regimes, Pressures, Atmospheres & Ratios
# ---------------------------------------------------------------------------
PROCESS_OPERATIONS: List[str] = [
    # Nuclear / Atomic
    "nucleosynthesis_fuse",
    "ionize_bind",
    # Chemical & Thermal
    "mix",
    "dissolve",
    "heat",
    "boil_reflux",
    "calcine",
    "roast_oxidize",
    "combust_burn",
    "distill_fractionate",
    "crystallize_precipitate",
    "electrolyze",
    "catalytic_synthesize",
    "polymerize",
    # Metallurgical & Materials
    "smelt_reduce",
    "alloy_melt",
    "quench_temper",
    "sinter_fire",
    "dope_diffuse",
    "draw_extrude",
    "coat_plate",
    # Mechanical & Engineering
    "forge_shape",
    "assemble_mechanically",
    "wire_circuit",
    "seal_encapsulate",
]
NUM_OPERATIONS = len(PROCESS_OPERATIONS)
OP_TO_IDX: Dict[str, int] = {op: i for i, op in enumerate(PROCESS_OPERATIONS)}

# Second step can also be "none" (single-step recipe)
STEP2_OPERATIONS: List[str] = ["none"] + PROCESS_OPERATIONS
STEP2_TO_IDX: Dict[str, int] = {op: i for i, op in enumerate(STEP2_OPERATIONS)}

# 7 Temperature Regimes (representative °C and description)
TEMP_REGIMES: List[Tuple[str, float, str]] = [
    ("cryogenic", -196.0, "Cryogenic cooling (-196°C)"),
    ("ambient", 25.0, "Room temperature (25°C)"),
    ("boil_reflux", 100.0, "Boiling / aqueous reflux (100°C)"),
    ("moderate_thermal", 350.0, "Moderate thermal / hydrothermal (350°C)"),
    ("calcination_red_heat", 850.0, "Calcination / red heat (850°C)"),
    ("smelting_white_heat", 1500.0, "Smelting / metallurgical melt (1500°C)"),
    ("plasma_arc_fusion", 5000.0, "Electric arc / plasma / nuclear fusion (>5000°C)"),
]
NUM_TEMP_REGIMES = len(TEMP_REGIMES)

# 4 Pressure Regimes
PRESSURE_REGIMES: List[Tuple[str, float, str]] = [
    ("vacuum", 0.01, "Vacuum (0.01 atm)"),
    ("ambient", 1.0, "Atmospheric pressure (1.0 atm)"),
    ("high_pressure", 50.0, "High pressure autoclave/reactor (50 atm)"),
    ("extreme_pressure", 500.0, "Extreme synthesis pressure (500 atm)"),
]
NUM_PRESSURE_REGIMES = len(PRESSURE_REGIMES)

# 5 Atmosphere / Medium Regimes
ATMOSPHERE_REGIMES: List[str] = [
    "inert_argon",
    "oxidizing_air",
    "reducing_h2_co",
    "aqueous_solution",
    "vacuum",
]
NUM_ATMOSPHERES = len(ATMOSPHERE_REGIMES)
ATMOS_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ATMOSPHERE_REGIMES)}

# 7 External Physics Excitation / Field / Radiation Regimes
EXCITATION_FIELDS: List[str] = [
    "none",
    "dc_voltage_bias",          # Applied DC voltage / electric current (anodization, electroluminescence, charge injection)
    "high_voltage_discharge",   # High-voltage electric field / arc / vacuum discharge (thermionic emission, dielectric poling)
    "uv_solar_light",           # UV / visible photonic illumination (hν >= E_g: photovoltaics, photocatalysis, photo-oxidation)
    "ionizing_radiation_beam",  # Neutron flux / electron beam / proton irradiation (Neutron Transmutation Doping 30Si(n,g)31P, defect engineering)
    "magnetic_induction_field", # External B-field / RF induction (permanent magnetic dipole poling, Hall sensing, eddy induction)
    "ultrasonic_acoustic",      # High-intensity ultrasonic / mechanical stress wave (sonochemistry, cavitation, piezoresistance)
]
NUM_EXCITATION_FIELDS = len(EXCITATION_FIELDS)
EXCITE_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(EXCITATION_FIELDS)}

# 7 Stoichiometric / Component Ratio Templates
STOICHIOMETRY_TEMPLATES: List[Tuple[str, List[int]]] = [
    ("1:1", [1, 1, 1, 1]),
    ("2:1", [2, 1, 1, 1]),
    ("1:2", [1, 2, 1, 1]),
    ("2:3", [2, 3, 1, 1]),
    ("3:1", [3, 1, 1, 1]),
    ("1:3", [1, 3, 1, 1]),
    ("catalytic_10:1", [10, 1, 1, 1]),
]
NUM_STOICH = len(STOICHIOMETRY_TEMPLATES)

INGREDIENT_ROLES: List[str] = [
    "primary_reactant",
    "secondary_reactant",
    "catalyst",
    "solvent_medium",
    "reducing_agent",
    "oxidizer",
    "dopant",
    "structural_part",
    "functional_core",
    "excitation_source",
    "active_medium",
]

# Core tracked elements for lineage fingerprint (14 dims)
TRACKED_ELEMENTS: List[str] = [
    "H", "C", "O", "N", "S", "Si", "Fe", "Cu", "Al", "Na", "Cl", "Ca", "Li", "Other"
]
NUM_TRACKED_ELEMENTS = len(TRACKED_ELEMENTS)


def normalize_item_name(name: str) -> str:
    """Normalize an item name for canonical deduplication."""
    s = name.strip().lower()
    # Remove parenthetical chemical symbol suffix like "hydrogen (h)" -> "hydrogen"
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown_item"


def extract_concept_family(name: str, category: str, capabilities: List[str]) -> str:
    """Group items into functional/chemical concept families to prevent mode collapse."""
    low = name.lower()
    if any(k in low for k in ("motor", "actuator", "solenoid")):
        return "electromagnetic_actuator_motor"
    if any(k in low for k in ("core", "transformer", "inductor")):
        return "electromagnetic_core"
    if any(k in low for k in ("battery", "voltaic", "cell", "accumulator")):
        return "electrochemical_cell"
    if any(k in low for k in ("furnace", "heater", "heating", "kiln")):
        return "thermal_furnace_heater"
    if any(k in low for k in ("diode", "transistor", "junction", "wafer", "doped")):
        return "semiconductor_electronics"
    if any(k in low for k in ("glass", "lens", "optical", "fused_silica")):
        return "optical_glass"
    if any(k in low for k in ("cement", "mortar", "concrete", "binder")):
        return "cement_binder"
    if any(k in low for k in ("wire", "cable", "electrode", "busbar", "anode", "cathode")):
        return "conductive_wire_electrode"
    if any(k in low for k in ("crucible", "vessel", "tank", "reactor")):
        return "containment_vessel"
    if "chloride" in low:
        return "chloride_compound"
    if "oxide" in low or "alumina" in low or "silica" in low or "quicklime" in low:
        return "oxide_compound"
    if "sulfide" in low or "sulfate" in low or "sulfuric" in low or "sulfur" in low:
        return "sulfur_compound"
    if "hydroxide" in low or "ammonia" in low:
        return "alkaline_base"
    if "acid" in low:
        return "acid_compound"
    if "water" in low or "peroxide" in low:
        return "aqueous_solvent"
    if category in ("alloy", "composite"):
        return f"{category}_{'_'.join(sorted(capabilities[:2]))}"
    return f"{category}_{normalize_item_name(name)[:15]}"


def default_properties() -> Dict[str, float]:
    return {k: 0.2 for k in PROPERTY_KEYS}


def clamp_properties(props: Dict[str, Any]) -> Dict[str, float]:
    out = default_properties()
    if not isinstance(props, dict):
        return out
    for k in PROPERTY_KEYS:
        if k in props:
            try:
                val = float(props[k])
                out[k] = max(0.0, min(1.0, round(val, 4)))
            except (ValueError, TypeError):
                pass
    return out


def sanitize_capabilities(caps: List[Any]) -> List[str]:
    if not isinstance(caps, list):
        return []
    valid = []
    for c in caps:
        cs = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if cs in CAPABILITY_TO_IDX and cs not in valid:
            valid.append(cs)
    return valid


def derive_real_world_utility(
    name: str,
    formula: str,
    tier: int,
    category: str,
    capabilities: List[str],
    description: str = "",
) -> Dict[str, Any]:
    """Translate an invented chemical, material, component, or device into its concrete real-world useful product equivalent and applications."""
    low = name.lower()
    caps = set(capabilities or [])

    if "electromagnetic actuator" in low or ("actuator" in low and "cermet" in low):
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "Abrasive-Duty Industrial Solenoid Valve & Geotechnical Actuator",
            "practical_applications": [
                "High-pressure mining & mineral slurry control valves",
                "Downhole geothermal & oilfield flow-control actuators",
                "Erosion-resistant industrial automation solenoids",
            ],
            "real_world_interpretation": "A complete electromechanical linear actuator whose Fe3Si-Al2O3 cermet body channels magnetic flux from dual copper solenoids while resisting extreme abrasive slurry and chemical wear.",
        }
    if "photovoltaic" in low or "solar cell" in low or "photodiode" in low or "photodetector" in low:
        return {
            "utility_score": 0.96,
            "real_world_equivalent": "Solid-State Photovoltaic Solar Cell & Optical Photodiode Detector",
            "practical_applications": [
                "Direct solar-to-DC electrical power generation",
                "Near-IR & visible optical communication photodetectors",
                "Self-powered industrial light & flame radiation sensors",
            ],
            "real_world_interpretation": "Incident photons (hν >= E_g) excite electron-hole pairs across the semiconductor heterojunction/p-n depletion zone, generating a continuous DC photocurrent.",
        }
    if "photoelectrochemical" in low or "photocatalytic" in low or "solar hydrogen" in low:
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "Photoelectrochemical Solar Water-Splitting Hydrogen Reactor",
            "practical_applications": [
                "Direct solar-driven green hydrogen (H2) production from water",
                "Photocatalytic wastewater oxidation & purification",
                "Off-grid solar chemical fuel synthesis",
            ],
            "real_world_interpretation": "Uses photonic illumination (hν) on a semiconductor/oxide photoelectrode immersed in water to split H2O directly into hydrogen and oxygen via photogenerated charge carriers.",
        }
    if "light-emitting" in low or "electroluminescent" in low or "led" in low.split():
        return {
            "utility_score": 0.94,
            "real_world_equivalent": "Solid-State Silicon Carbide Light-Emitting Diode (SiC Optoelectronic Emitter)",
            "practical_applications": [
                "High-temperature solid-state status indicators & optical emitters",
                "Harsh-environment UV/blue optoelectronic signaling",
                "Radiation-hardened solid-state illumination",
            ],
            "real_world_interpretation": "Applying forward DC voltage bias injects electrons and holes into the wide-bandgap semiconductor junction, producing radiative recombination photons (electroluminescence).",
        }
    if "anodized" in low:
        return {
            "utility_score": 0.92,
            "real_world_equivalent": "Anodized Aluminum Dielectric Passivation Substrate & Capacitor Foil",
            "practical_applications": [
                "Electrolytic capacitor dielectric barrier foils",
                "Insulated metal substrates (IMS) for high-power electronics cooling",
                "Corrosion- and wear-resistant architectural & aerospace housings",
            ],
            "real_world_interpretation": "Electrochemical anodic voltage bias grows a dense, adherent nanoporous/barrier Al2O3 dielectric film directly from the parent aluminum surface.",
        }
    if "transmutation" in low or "phosphorus-doped" in low or "ntd" in low:
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "Neutron-Transmutation-Doped N-Type Power Semiconductor Wafer (NTD-Si)",
            "practical_applications": [
                "High-voltage HVDC thyristors & IGBT power grid switches",
                "Ultra-uniform n-type substrates for particle & radiation detectors",
                "Complementary n-type partner for p-n diodes and solar cells",
            ],
            "real_world_interpretation": "Thermal neutron capture on 30Si nuclei (30Si + n0 -> 31Si -> 31P + β-) transmutes silicon atoms into ultra-uniform substitutional phosphorus donors across the crystal bulk.",
        }
    if "permanent magnet" in low or "magnetically poled" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Magnetically Poled Permanent Magnet Rotor & Field Pole",
            "practical_applications": [
                "Permanent-magnet DC motors & synchronous electric generators",
                "Magnetic Hall-effect position & speed sensor targets",
                "Magnetic latches, separators, and acoustic transducers",
            ],
            "real_world_interpretation": "External magnetic field poling during thermal treatment aligns ferromagnetic domains along the easy magnetization axis, locking in remanent magnetic flux (B_r).",
        }
    if "thermionic" in low or "electron emitter" in low or "cathode ray" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Vacuum Thermionic Electron Gun & E-Beam Cathode Source",
            "practical_applications": [
                "Electron-beam welding, melting, and physical vapor deposition",
                "X-ray tube & vacuum microwave amplifier cathodes",
                "Scanning electron microscopy & electron lithography sources",
            ],
            "real_world_interpretation": "High-temperature thermal excitation combined with a high-voltage electric field in vacuum overcomes the material work function (Φ), emitting a focused electron beam.",
        }
    if "transformer" in low:
        return {
            "utility_score": 0.96,
            "real_world_equivalent": "AC Electrical Power Step-Up / Step-Down Transformer",
            "practical_applications": [
                "Electrical grid voltage conversion & distribution",
                "Galvanic isolation in industrial power supplies",
                "Impedance matching in high-power induction equipment",
            ],
            "real_world_interpretation": "Magnetically coupled dual-winding copper coils on a low-loss ferrosilicon/silicon-steel core that steps AC voltage up or down with high electromagnetic efficiency.",
        }
    if "motor" in low or "relay" in low:
        return {
            "utility_score": 0.96,
            "real_world_equivalent": "Industrial Electromechanical Motor / High-Current Contactor Relay",
            "practical_applications": [
                "Automated mechanical drive systems & pumps",
                "High-current industrial circuit switching (contactors)",
                "Precision robotic & valve positioning",
            ],
            "real_world_interpretation": "Converts electrical current into mechanical torque or linear force via electromagnetic attraction between copper-wound stator coils and a ferromagnetic armature.",
        }
    if "furnace" in low or "kiln" in low:
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "High-Temperature Electric Resistance Box / Melt Furnace",
            "practical_applications": [
                "Metallurgical alloy melting & heat treatment up to 1500°C",
                "Ceramic & cermet powder sintering",
                "Semiconductor wafer thermal diffusion & annealing",
            ],
            "real_world_interpretation": "Combines high-temperature SiC/intermetallic heating elements inside an insulated alumina/calcium-aluminate refractory chamber to deliver controlled industrial process heat.",
        }
    if "electrolyzer" in low or "electrolysis" in low:
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "Industrial Alkaline Water Electrolyzer / Chlor-Alkali Cell",
            "practical_applications": [
                "Green hydrogen & oxygen gas production from water",
                "Electrochemical chlorine & caustic soda manufacturing",
                "Grid-scale chemical energy storage",
            ],
            "real_world_interpretation": "An electrochemical reactor cell using conductive electrodes and an ionic electrolyte to split water or brine into clean hydrogen fuel and industrial chemical feedstocks.",
        }
    if "battery" in low or "voltaic" in low or "galvanic" in low or ("cell" in low and "energy_storage" in caps):
        return {
            "utility_score": 0.96,
            "real_world_equivalent": "Primary / Rechargeable Electrochemical Battery Cell",
            "practical_applications": [
                "Portable & backup DC electrical power storage",
                "Off-grid sensor & actuator power supply",
                "High-energy-density molten-salt or alkaline reserve batteries",
            ],
            "real_world_interpretation": "Converts stored chemical free energy directly into DC electrical voltage across an electropositive metal anode, ionic electrolyte, and conductive cathode.",
        }
    if "solar" in low or "photovoltaic" in low or "pn-junction" in low or "p-n junction" in low or "diode" in low:
        return {
            "utility_score": 0.96,
            "real_world_equivalent": "Silicon PN-Junction Rectifier Diode & Photovoltaic Solar Cell",
            "practical_applications": [
                "Photovoltaic solar electricity generation",
                "AC-to-DC power supply rectification",
                "Solid-state optical light sensing (photodiodes)",
            ],
            "real_world_interpretation": "A monolithic p-type/n-type semiconductor junction with built-in electric field that rectifies alternating current and converts sunlight directly into DC electrical power.",
        }
    if "capacitor" in low or "mos " in low or "gate stack" in low or "transistor" in low:
        return {
            "utility_score": 0.94,
            "real_world_equivalent": "MOSFET Gate Stack / Metal-Oxide-Semiconductor Capacitor",
            "practical_applications": [
                "Digital logic gates & microprocessor switching",
                "Power electronics voltage regulation",
                "Capacitive charge storage & signal filtering",
            ],
            "real_world_interpretation": "Foundational building block of modern integrated circuits, using an oxide dielectric layer between a conductive metal gate and doped silicon substrate to control charge carriers.",
        }
    if "thermocouple" in low or "thermoelectric" in low or ("heterojunction" in low and "sensor" in caps):
        return {
            "utility_score": 0.91,
            "real_world_equivalent": "Industrial Thermocouple Probe & Seebeck Thermoelectric Generator",
            "practical_applications": [
                "High-temperature furnace & turbine thermal sensing",
                "Solid-state waste-heat-to-electricity harvesting",
                "Closed-loop industrial process temperature control",
            ],
            "real_world_interpretation": "Exploits the Seebeck effect across a junction of dissimilar conductors or doped silicide-silicon semiconductors to generate a measurable voltage proportional to temperature gradients.",
        }
    if "solenoid" in low or "electromagnet" in low or "inductor" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Industrial Solenoid Electromagnet & High-Current Inductor Coil",
            "practical_applications": [
                "Electromagnetic valves, brakes, and clutches",
                "Magnetic plasma confinement & induction heating coils",
                "Power supply LC filtering & magnetic lifting",
            ],
            "real_world_interpretation": "Concentrates magnetic flux inside a high-permeability iron-silicon core wound with conductive copper turns, enabling electrical-to-magnetic force conversion and inductive energy storage.",
        }
    if "heating element" in low or "heater" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Globar SiC / Ceramic-Insulated Industrial Furnace Heating Element",
            "practical_applications": [
                "1400–1600°C electric laboratory & industrial furnaces",
                "Semiconductor diffusion & ceramic sintering kilns",
                "Corrosion-resistant radiant process heaters",
            ],
            "real_world_interpretation": "Converts electrical power into intense radiative and conductive heat via Joule heating in a refractory silicon carbide or intermetallic core electrically isolated by alumina ceramic.",
        }
    if "wafer" in low or ("doped" in low and "silicon" in low):
        return {
            "utility_score": 0.90,
            "real_world_equivalent": "Prime Doped Semiconductor Silicon Wafer (p-Type / n-Type)",
            "practical_applications": [
                "Substrate for solar cells, diodes, and CMOS transistors",
                "Solid-state pressure & temperature micro-sensors (MEMS)",
                "Power rectifier & thyristor fabrication",
            ],
            "real_world_interpretation": "Extrinsic crystalline silicon engineered with controlled acceptor (Al) or donor (Li) carrier concentrations to serve as the active semiconductor substrate for electronics.",
        }
    if "lithium deuteride" in low:
        return {
            "utility_score": 0.92,
            "real_world_equivalent": "Solid-State Deuterium Isotope Storage Salt & Fusion Fuel Pellet",
            "practical_applications": [
                "Compact solid-state deuterium fuel for fusion energy reactors",
                "High-density heavy-hydrogen isotope storage & transport",
                "Neutron-shielding & nuclear research targets",
            ],
            "real_world_interpretation": "Stores heavy hydrogen (deuterium) at high volumetric density inside a stable ionic crystal lattice (Li+ D-) without requiring cryogenic high-pressure gas cylinders.",
        }
    if "deuterium" in low:
        return {
            "utility_score": 0.89,
            "real_world_equivalent": "Heavy Hydrogen Isotope (Deuterium / 2H) Nuclear & Tracer Feedstock",
            "practical_applications": [
                "D-T and D-D fusion energy fuel",
                "Heavy water (D2O) neutron moderator synthesis",
                "Deuterated pharmaceuticals & NMR spectroscopy solvents",
            ],
            "real_world_interpretation": "Stable heavy isotope of hydrogen containing one proton and one neutron, essential for clean fusion energy, neutron moderation, and isotopic tracing.",
        }
    if "water" in low and tier == 2:
        return {
            "utility_score": 0.98,
            "real_world_equivalent": "Purified Process Water & Universal Polar Solvent (H2O)",
            "practical_applications": [
                "Universal solvent for aqueous acid/base/salt chemistry",
                "Feedstock for green hydrogen electrolysis",
                "Hydraulic cement curing & industrial heat-transfer coolant",
            ],
            "real_world_interpretation": "Foundational polar solvent and chemical reagent enabling wet chemistry, acid/base synthesis, cement hydration, and thermal cooling.",
        }
    if "sodium hydroxide" in low or "caustic" in low:
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "Industrial Caustic Soda / Lye (NaOH)",
            "practical_applications": [
                "Alkaline water electrolysis & battery electrolytes",
                "Bauxite alumina refining (Bayer process) & paper pulping",
                "Chemical neutralization, soap & detergent manufacturing",
            ],
            "real_world_interpretation": "Workhorse industrial strong base used worldwide for chemical manufacturing, ore refining, water treatment, and alkaline electrochemical cells.",
        }
    if "calcium hydroxide" in low or "slaked lime" in low:
        return {
            "utility_score": 0.92,
            "real_world_equivalent": "Commercial Hydrated / Slaked Lime (Ca(OH)2)",
            "practical_applications": [
                "Structural lime mortar, plaster & pozzolanic concrete",
                "Flue-gas desulfurization & industrial acid neutralization",
                "Water clarification & soil stabilization",
            ],
            "real_world_interpretation": "Versatile inorganic base produced by hydrating quicklime, widely used in construction mortars, environmental acid scrubbing, and water treatment.",
        }
    if "lithium hydroxide" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Battery-Grade Lithium Hydroxide & CO2 Scrubber (LiOH)",
            "practical_applications": [
                "Precursor for high-nickel EV lithium-ion battery cathodes",
                "Spacecraft & submarine life-support CO2 gas scrubbers",
                "High-temperature lithium lubricating greases",
            ],
            "real_world_interpretation": "Critical lithium chemical feedstock used to manufacture electric-vehicle battery cathodes and absorb carbon dioxide in enclosed life-support systems.",
        }
    if "ammonia" in low:
        return {
            "utility_score": 0.97,
            "real_world_equivalent": "Anhydrous Ammonia Fertilizer & Carbon-Free Fuel (NH3)",
            "practical_applications": [
                "Global agricultural nitrogen fertilizer production",
                "Carbon-free hydrogen energy carrier & marine fuel",
                "Industrial refrigeration (R-717) & nitric acid synthesis",
            ],
            "real_world_interpretation": "One of civilization's most essential industrial chemicals, sustaining global food production via nitrogen fertilizers and serving as a high-density carbon-free fuel.",
        }
    if "methane" in low:
        return {
            "utility_score": 0.94,
            "real_world_equivalent": "Synthetic Natural Gas (SNG / CH4) Clean Fuel",
            "practical_applications": [
                "High-efficiency gas turbine electricity generation",
                "Industrial & residential thermal heating fuel",
                "Chemical feedstock for synthesis gas & methanol",
            ],
            "real_world_interpretation": "Clean-burning high-energy-density hydrocarbon gas used directly for power generation, high-temperature heating, and chemical synthesis.",
        }
    if "hydrochloric" in low or "hydrogen chloride" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Industrial Hydrochloric / Muriatic Acid (HCl)",
            "practical_applications": [
                "Steel pickling & metal oxide scale removal",
                "Semiconductorgrade trichlorosilane etching & purification",
                "Ore leaching, pH control & chloride salt production",
            ],
            "real_world_interpretation": "Essential strong mineral acid used across metallurgy, semiconductor wafer cleaning, and chemical processing.",
        }
    if "sulfuric" in low or "sulfurous" in low or "sulfur dioxide" in low:
        return {
            "utility_score": 0.92,
            "real_world_equivalent": "Industrial Sulfur Chemical / Acid Feedstock",
            "practical_applications": [
                "Sulfuric acid & sulfate fertilizer production",
                "Metallurgical ore leaching & chemical processing",
                "Chemical reducing & bleaching agent",
            ],
            "real_world_interpretation": "Core sulfur intermediate for industrial acid manufacturing, mineral processing, and chemical synthesis.",
        }
    if "cement" in low or "mortar" in low or "concrete" in low or "calcium aluminate" in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Castable Refractory Cement & Structural Hydraulic Concrete",
            "practical_applications": [
                "Monolithic linings for steelmaking & smelting furnaces",
                "Rapid-hardening corrosion-resistant structural concrete",
                "High-temperature kiln &foundry castables",
            ],
            "real_world_interpretation": "Hydraulic ceramic binder that reacts with water or sinters into a high-strength, heat-resistant structural concrete for infrastructure and industrial furnaces.",
        }
    if "glass" in low:
        return {
            "utility_score": 0.92,
            "real_world_equivalent": "Commercial Optical & Architectural Soda-Lime / Silicate Glass",
            "practical_applications": [
                "Transparent windows, optical lenses & solar cell covers",
                "Chemically inert laboratory glassware & reaction vessels",
                "High-voltage electrical & thermal glazing insulation",
            ],
            "real_world_interpretation": "Optically transparent, chemically inert amorphous silicate glass network used for windows, optics, solar module encapsulation, and chemical vessels.",
        }
    if "silicon carbide" in low and "cermet" not in low and "reinforced" not in low:
        return {
            "utility_score": 0.93,
            "real_world_equivalent": "Industrial Carborundum (SiC) Super-Abrasive & Power Semiconductor",
            "practical_applications": [
                "Diamond-like cutting, grinding, and machining abrasives",
                "High-voltage EV inverter wide-bandgap semiconductors",
                "1600°C furnace refractories, armor & mechanical seals",
            ],
            "real_world_interpretation": "Ultra-hard covalent ceramic combining extreme thermal shock resistance, abrasive cutting power, and wide-bandgap semiconductor performance.",
        }
    if "cermet" in low or "reinforced" in low:
        return {
            "utility_score": 0.89,
            "real_world_equivalent": "Wear-Resistant Metal-Matrix Composite (MMC / Cermet) Tooling & Armor",
            "practical_applications": [
                "High-speed cutting tool inserts & mining drill bits",
                "Lightweight aerospace & automotive brake rotors",
                "High-temperature erosion-resistant turbine & valve shields",
            ],
            "real_world_interpretation": "Combines the fracture toughness and thermal conductivity of a metallic/intermetallic matrix with the extreme hardness and refractory stability of ceramic particles.",
        }
    if "silicon steel" in low:
        return {
            "utility_score": 0.94,
            "real_world_equivalent": "Electrical Transformer & EV Motor Lamination Steel (Fe-Si)",
            "practical_applications": [
                "Grid power transformer magnetic cores",
                "Electric vehicle traction motor stators & rotors",
                "High-efficiency industrial generators & inductors",
            ],
            "real_world_interpretation": "Soft ferromagnetic iron-silicon alloy whose high electrical resistivity suppresses eddy-current core losses in AC transformers and electric motors.",
        }
    if "carbon steel" in low or "cast iron" in low:
        return {
            "utility_score": 0.95,
            "real_world_equivalent": "Structural Carbon Steel & Heavy Machinery Cast Iron",
            "practical_applications": [
                "Load-bearing building frames, bridges & pressure vessels",
                "Machine tool beds, engine blocks & pump housings",
                "Industrial shafts, gears, fasteners & cutting tools",
            ],
            "real_world_interpretation": "Foundational structural ferrous alloy of modern engineering, delivering high tensile strength, machinability, and toughness.",
        }
    if "bronze" in low or ("copper" in low and "aluminum" in low):
        return {
            "utility_score": 0.91,
            "real_world_equivalent": "Marine-Grade Aluminum/Silicon Bronze Structural & Bearing Alloy",
            "practical_applications": [
                "Seawater-resistant ship propellers, pumps & naval valves",
                "Heavy-load sleeve bearings, landing gear bushings & gears",
                "Non-sparking petrochemical safety tools & electrical connectors",
            ],
            "real_world_interpretation": "High-strength, corrosion-resistant copper alloy fortified by a self-healing alumina/silica surface film for marine, aerospace, and bearing hardware.",
        }
    if "aluminum oxide" in low:
        return {
            "utility_score": 0.94,
            "real_world_equivalent": "Technical Corundum / Alumina Ceramic (Al2O3) & Smelting Feedstock",
            "practical_applications": [
                "High-temperature metallurgical melting crucibles & furnace tubes",
                "High-voltage electrical insulators & IC ceramic substrates",
                "Hall-Héroult aluminum metal smelting feedstock & grinding media",
            ],
            "real_world_interpretation": "Refractory oxide ceramic with a 2072°C melting point and high dielectric strength, used for crucibles, electrical insulation, and aluminum metal production.",
        }
    if "sodium chloride" in low or "lithium chloride" in low or "calcium chloride" in low:
        return {
            "utility_score": 0.91,
            "real_world_equivalent": "Industrial Halide Electrolyte & Molten-Salt Thermal/Chemical Feedstock",
            "practical_applications": [
                "Molten-salt & aqueous electrolysis for metal/chlorine recovery",
                "Thermal energy storage & heat-transfer molten salt baths",
                "De-icing, desiccant drying & battery electrolyte synthesis",
            ],
            "real_world_interpretation": "Ionic halide salt that conducts electricity when dissolved in water or melted, serving as the core electrolyte for industrial electrochemistry.",
        }

    # Tier-based fallback for any other item
    if tier >= 5:
        return {
            "utility_score": 0.92,
            "real_world_equivalent": f"Engineered Functional Machine / Device ({name})",
            "practical_applications": [
                "Automated industrial actuation, power conversion, or thermal processing",
                "Integrated electromechanical or electrochemical system operation",
                "Modular engineering subsystem deployment",
            ],
            "real_world_interpretation": f"Multi-component engineered device ({formula}) integrating active electrical, magnetic, thermal, or chemical components into a practical working machine.",
        }
    if tier == 4:
        return {
            "utility_score": 0.88,
            "real_world_equivalent": f"Functional Engineering Component ({name})",
            "practical_applications": [
                "Sub-assembly for electric motors, furnaces, sensors, or power systems",
                "Solid-state electronic or electromagnetic circuit integration",
                "Industrial instrumentation & control hardware",
            ],
            "real_world_interpretation": f"Fabricated functional component ({formula}) ready to be wired or assembled into higher-level electrical, thermal, or mechanical machines.",
        }
    if tier == 3:
        return {
            "utility_score": 0.84,
            "real_world_equivalent": f"Engineered Structural / Functional Material ({name})",
            "practical_applications": [
                "High-temperature structural hardware & wear-resistant tooling",
                "Electrical, magnetic, or refractory device fabrication",
                "Chemical reactor & furnace construction",
            ],
            "real_world_interpretation": f"Synthesized engineering material ({formula}) offering tailored mechanical, thermal, or electromagnetic properties for component manufacturing.",
        }
    if tier == 2:
        return {
            "utility_score": 0.82,
            "real_world_equivalent": f"Industrial Chemical Feedstock & Reagent ({name})",
            "practical_applications": [
                "Precursor for ceramics, glasses, cements, and metallurgical smelting",
                "Electrochemical, catalytic, or aqueous chemical synthesis",
                "Industrial materials processing & surface treatment",
            ],
            "real_world_interpretation": f"Refined chemical compound ({formula}) serving as a reactive building block for advanced materials, electrolytes, and industrial processing.",
        }
    return {
        "utility_score": 0.80,
        "real_world_equivalent": f"Fundamental Elemental / Atomic Feedstock ({name})",
        "practical_applications": [
            "Primary feedstock for chemical synthesis & alloy metallurgy",
            "Semiconductor doping, catalysis, or nuclear/plasma physics",
            "Electrochemical electrode & energy material fabrication",
        ],
        "real_world_interpretation": f"Foundational elemental or subatomic building block ({formula}) used to synthesize higher-tier compounds, alloys, and functional devices.",
    }


# ---------------------------------------------------------------------------
# 6. Core Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class LibraryItem:
    item_id: str
    name: str
    formula_or_spec: str
    tier: int
    category: str
    phase: str
    description: str
    properties: Dict[str, float] = field(default_factory=default_properties)
    capabilities: List[str] = field(default_factory=list)
    elemental_lineage: Dict[str, float] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    created_by_blueprint_id: Optional[str] = None
    discovered_by_worker: str = "seed"
    discovery_step: int = 0
    times_used_in_valid_recipes: int = 0
    is_known_real_world: bool = True
    novelty_classification: str = "KNOWN_REAL_WORLD"  # SEED_PRIMITIVE | KNOWN_REAL_WORLD | NOVEL_FRONTIER
    novelty_rationale: str = ""
    escalation_status: str = "NOT_REQUIRED"  # NOT_REQUIRED | PENDING_ESCALATION | VERIFIED_NOVEL_BREAKTHROUGH | PLAUSIBLE_FRONTIER_CONSTRUCT | RECLASSIFIED_KNOWN | REJECTED_DEEP_AUDIT
    recipe_summary: str = ""
    judge_votes: Dict[str, Any] = field(default_factory=dict)
    deep_audit: Dict[str, Any] = field(default_factory=dict)
    utility_score: float = 0.80
    real_world_equivalent: str = ""
    practical_applications: List[str] = field(default_factory=list)
    real_world_interpretation: str = ""
    simulation_verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LibraryItem":
        data = dict(data)
        data["properties"] = clamp_properties(data.get("properties", {}))
        data["capabilities"] = sanitize_capabilities(data.get("capabilities", []))
        if "novelty_classification" not in data:
            if data.get("discovered_by_worker") == "seed_library":
                data["novelty_classification"] = "SEED_PRIMITIVE"
            elif data.get("is_known_real_world", True):
                data["novelty_classification"] = "KNOWN_REAL_WORLD"
            else:
                data["novelty_classification"] = "NOVEL_FRONTIER"
        if "escalation_status" not in data:
            data["escalation_status"] = (
                "PENDING_ESCALATION"
                if data["novelty_classification"] == "NOVEL_FRONTIER"
                else "NOT_REQUIRED"
            )
        if not data.get("real_world_equivalent") or not data.get("practical_applications"):
            prof = derive_real_world_utility(
                name=str(data.get("name", "")),
                formula=str(data.get("formula_or_spec", "")),
                tier=int(data.get("tier", 1)),
                category=str(data.get("category", "compound")),
                capabilities=data["capabilities"],
                description=str(data.get("description", "")),
            )
            if not data.get("utility_score"):
                data["utility_score"] = prof["utility_score"]
            if not data.get("real_world_equivalent"):
                data["real_world_equivalent"] = prof["real_world_equivalent"]
            if not data.get("practical_applications"):
                data["practical_applications"] = prof["practical_applications"]
            if not data.get("real_world_interpretation"):
                data["real_world_interpretation"] = prof["real_world_interpretation"]
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def property_vector(self) -> List[float]:
        return [float(self.properties.get(k, 0.2)) for k in PROPERTY_KEYS]

    def capability_vector(self) -> List[float]:
        vec = [0.0] * NUM_CAPABILITIES
        for c in self.capabilities:
            if c in CAPABILITY_TO_IDX:
                vec[CAPABILITY_TO_IDX[c]] = 1.0
        return vec


@dataclass
class BlueprintIngredient:
    item_id: str
    item_name: str
    formula: str
    ratio: int = 1
    role: str = "primary_reactant"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessStep:
    step_order: int
    operation: str
    temperature_c: float
    pressure_atm: float
    atmosphere: str
    duration: str = "15 min"
    details: str = ""
    excitation_field: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NeuralPrediction:
    predicted_validity: float = 0.5
    predicted_compatibility: float = 0.5
    predicted_physics_score: float = 0.5
    predicted_utility: float = 0.75
    predicted_properties: Dict[str, float] = field(default_factory=default_properties)
    predicted_capabilities: List[str] = field(default_factory=list)
    expected_reward: float = 0.0
    policy_indices: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Blueprint:
    blueprint_id: str
    proposer_id: str
    strategy: str
    target_goal: str
    ingredients: List[BlueprintIngredient]
    steps: List[ProcessStep]
    neural_prediction: NeuralPrediction = field(default_factory=NeuralPrediction)
    equipment_used: List[str] = field(default_factory=list)

    @staticmethod
    def compute_canonical_id(
        ingredients: List[BlueprintIngredient],
        steps: List[ProcessStep],
    ) -> str:
        """Compute deterministic hash from sorted ingredients + process steps."""
        ing_part = sorted(
            f"{ing.item_id}:{ing.ratio}:{ing.role}" for ing in ingredients
        )
        step_part = [
            f"{s.operation}:{int(round(s.temperature_c))}:{round(s.pressure_atm, 2)}:{s.atmosphere}"
            + (f":{s.excitation_field}" if getattr(s, "excitation_field", "none") not in ("", "none") else "")
            for s in steps
        ]
        payload = "|".join(ing_part) + "||" + "->".join(step_part)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"bp_{digest}"

    def summary_recipe_str(self) -> str:
        ing_str = " + ".join(
            f"{ing.ratio}x {ing.item_name} ({ing.role})" for ing in self.ingredients
        )
        step_str = " -> ".join(
            f"{s.operation}@{int(s.temperature_c)}°C/{s.pressure_atm}atm[{s.atmosphere}"
            + (f"|⚡{s.excitation_field}]" if getattr(s, "excitation_field", "none") not in ("", "none") else "]")
            for s in self.steps
        )
        eq_str = f" | Equip: {', '.join(self.equipment_used)}" if self.equipment_used else ""
        return f"[{ing_str}] ==> [{step_str}]{eq_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "proposer_id": self.proposer_id,
            "strategy": self.strategy,
            "target_goal": self.target_goal,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "steps": [s.to_dict() for s in self.steps],
            "neural_prediction": self.neural_prediction.to_dict(),
            "equipment_used": list(self.equipment_used),
        }


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Blueprint":
        ings = [BlueprintIngredient(**i) for i in data.get("ingredients", [])]
        steps = [ProcessStep(**s) for s in data.get("steps", [])]
        np_dict = data.get("neural_prediction", {})
        n_pred = NeuralPrediction(
            predicted_validity=float(np_dict.get("predicted_validity", 0.5)),
            predicted_compatibility=float(np_dict.get("predicted_compatibility", 0.5)),
            predicted_physics_score=float(np_dict.get("predicted_physics_score", 0.5)),
            predicted_utility=float(np_dict.get("predicted_utility", 0.75)),
            predicted_properties=clamp_properties(np_dict.get("predicted_properties", {})),
            predicted_capabilities=sanitize_capabilities(np_dict.get("predicted_capabilities", [])),
            expected_reward=float(np_dict.get("expected_reward", 0.0)),
            policy_indices=dict(np_dict.get("policy_indices", {})),
        )
        return cls(
            blueprint_id=str(data.get("blueprint_id", "")),
            proposer_id=str(data.get("proposer_id", "")),
            strategy=str(data.get("strategy", "")),
            target_goal=str(data.get("target_goal", "")),
            ingredients=ings,
            steps=steps,
            neural_prediction=n_pred,
            equipment_used=list(data.get("equipment_used", [])),
        )


@dataclass
class JudgeVerdict:
    blueprint_id: str
    valid: bool
    physics_score: float  # 0.0 (nonsense) to 1.0 (textbook reaction/engineering)
    is_known_real_world: bool
    output_name: str
    formula_or_spec: str
    category: str
    tier: int
    phase: str
    description: str
    properties: Dict[str, float]
    capabilities: List[str]
    byproducts: List[str] = field(default_factory=list)
    failure_reason: str = ""
    repair_hint: Dict[str, Any] = field(default_factory=dict)
    judge_model: str = "multi_judge_panel(gemini+deepseek+glm)"
    latency_s: float = 0.0
    judge_votes: Dict[str, Any] = field(default_factory=dict)
    novelty_classification: str = "KNOWN_REAL_WORLD"  # KNOWN_REAL_WORLD | NOVEL_FRONTIER
    novelty_rationale: str = ""
    deep_audit: Dict[str, Any] = field(default_factory=dict)
    utility_score: float = 0.80
    real_world_equivalent: str = ""
    practical_applications: List[str] = field(default_factory=list)
    real_world_interpretation: str = ""
    simulation_verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeVerdict":
        data = dict(data)
        data["properties"] = clamp_properties(data.get("properties", {}))
        data["capabilities"] = sanitize_capabilities(data.get("capabilities", []))
        if "novelty_classification" not in data:
            data["novelty_classification"] = (
                "KNOWN_REAL_WORLD"
                if data.get("is_known_real_world", True)
                else "NOVEL_FRONTIER"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ExperienceRecord:
    blueprint: Blueprint
    verdict: JudgeVerdict
    reward: float
    is_new_discovery: bool
    discovered_item_id: Optional[str] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blueprint": self.blueprint.to_dict(),
            "verdict": self.verdict.to_dict(),
            "reward": self.reward,
            "is_new_discovery": self.is_new_discovery,
            "discovered_item_id": self.discovered_item_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceRecord":
        return cls(
            blueprint=Blueprint.from_dict(data.get("blueprint", {})),
            verdict=JudgeVerdict.from_dict(data.get("verdict", {})),
            reward=float(data.get("reward", 0.0)),
            is_new_discovery=bool(data.get("is_new_discovery", False)),
            discovered_item_id=data.get("discovered_item_id"),
            timestamp=float(data.get("timestamp", 0.0)),
        )

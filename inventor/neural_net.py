"""Joined PyTorch Neural Network (InventorNet & SharedBrain) running on CUDA (RTX 3090).

Learns:
1. What ingredients are compatible and synergistic (CompatibilityNet)
2. What recipe steps, temperatures, pressures, atmospheres & ratios work (RecipePolicyNet)
3. What physical properties (12D) and capabilities (24D) a blueprint will produce (ForwardWorldModel)
4. How to invert the model: given target desired properties, propose ingredients + recipe (SharedBrain)
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inventor.schemas import (
    ATMOS_TO_IDX,
    ATMOSPHERE_REGIMES,
    CAPABILITY_TO_IDX,
    CAPABILITY_VOCAB,
    NUM_ATMOSPHERES,
    NUM_CAPABILITIES,
    NUM_OPERATIONS,
    NUM_PHASES,
    NUM_PRESSURE_REGIMES,
    NUM_PROPERTIES,
    NUM_STOICH,
    NUM_TEMP_REGIMES,
    NUM_TRACKED_ELEMENTS,
    OP_TO_IDX,
    PHASE_TO_IDX,
    PRESSURE_REGIMES,
    PROCESS_OPERATIONS,
    PROPERTY_KEYS,
    STEP2_OPERATIONS,
    STEP2_TO_IDX,
    STOICHIOMETRY_TEMPLATES,
    TEMP_REGIMES,
    TRACKED_ELEMENTS,
    Blueprint,
    BlueprintIngredient,
    LibraryItem,
    NeuralPrediction,
    ProcessStep,
)

ITEM_FEAT_DIM = 128  # 12 props + 24 caps + 6 tier + 8 phase + 14 elem + 64 semantic trigram
GOAL_DIM = NUM_PROPERTIES + NUM_CAPABILITIES  # 12 + 24 = 36
MAX_INGREDIENTS = 3


def compute_trigram_hash_vector(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic character 3-gram signed feature hashing for open-ended item names/formulas."""
    vec = np.zeros(dim, dtype=np.float32)
    clean = f"^{text.lower().strip()}$"
    for i in range(max(1, len(clean) - 2)):
        trigram = clean[i : i + 3].encode("utf-8")
        h = int.from_bytes(hashlib.md5(trigram).digest()[:4], "little")
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def extract_item_feature_vector(item: LibraryItem) -> np.ndarray:
    """Convert any LibraryItem into a 128-D continuous feature vector."""
    props = np.array(
        [float(item.properties.get(k, 0.2)) for k in PROPERTY_KEYS], dtype=np.float32
    )
    caps = np.zeros(NUM_CAPABILITIES, dtype=np.float32)
    for c in item.capabilities:
        if c in CAPABILITY_TO_IDX:
            caps[CAPABILITY_TO_IDX[c]] = 1.0
    tier_vec = np.zeros(6, dtype=np.float32)
    t_idx = max(0, min(5, int(item.tier)))
    tier_vec[t_idx] = 1.0
    phase_vec = np.zeros(NUM_PHASES, dtype=np.float32)
    p_idx = PHASE_TO_IDX.get(item.phase.lower(), 3)
    phase_vec[p_idx] = 1.0
    elem_vec = np.zeros(NUM_TRACKED_ELEMENTS, dtype=np.float32)
    for i, el in enumerate(TRACKED_ELEMENTS):
        elem_vec[i] = float(item.elemental_lineage.get(el, 0.0))
    elem_norm = np.linalg.norm(elem_vec)
    if elem_norm > 1e-6:
        elem_vec /= elem_norm
    sem_text = f"{item.name} {item.formula_or_spec} {item.category}"
    sem_vec = compute_trigram_hash_vector(sem_text, dim=64)

    feat = np.concatenate([props, caps, tier_vec, phase_vec, elem_vec, sem_vec], axis=0)
    return feat


class InventorNet(nn.Module):
    """Multi-head neural network for ingredient compatibility, recipe synthesis, and property prediction."""

    def __init__(self, hidden_dim: int = 160):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Item Encoder
        self.item_encoder = nn.Sequential(
            nn.Linear(ITEM_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 2. Pairwise Bilinear & Self-Attention Compatibility Network
        self.pairwise_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.ing_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.compat_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 3. Goal-Conditioned Recipe Policy Network
        self.goal_encoder = nn.Sequential(
            nn.Linear(GOAL_DIM, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.policy_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.op1_head = nn.Linear(hidden_dim, NUM_OPERATIONS)
        self.op1_embed = nn.Embedding(NUM_OPERATIONS, 32)
        self.op2_head = nn.Linear(hidden_dim + 32, len(STEP2_OPERATIONS))
        self.temp_head = nn.Linear(hidden_dim + 32, NUM_TEMP_REGIMES)
        self.press_head = nn.Linear(hidden_dim + 32, NUM_PRESSURE_REGIMES)
        self.atmos_head = nn.Linear(hidden_dim + 32, NUM_ATMOSPHERES)
        self.stoich_head = nn.Linear(hidden_dim, NUM_STOICH)

        # 4. Recipe Encoder for Forward World Model
        recipe_raw_dim = (
            NUM_OPERATIONS
            + len(STEP2_OPERATIONS)
            + NUM_TEMP_REGIMES
            + NUM_PRESSURE_REGIMES
            + NUM_ATMOSPHERES
            + NUM_STOICH
        )
        self.recipe_encoder = nn.Sequential(
            nn.Linear(recipe_raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 5. Forward World Model
        self.world_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.physics_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.property_head = nn.Linear(hidden_dim, NUM_PROPERTIES)
        self.capability_head = nn.Linear(hidden_dim, NUM_CAPABILITIES)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def encode_ingredients(
        self, item_feats: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.item_encoder(item_feats)
        key_padding_mask = ~mask.bool()
        attn_out, _ = self.ing_attention(h, h, h, key_padding_mask=key_padding_mask)
        h = h + attn_out

        mask_f = mask.unsqueeze(-1).float()
        h_masked = h * mask_f
        count = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = h_masked.sum(dim=1) / count

        h0 = h[:, 0, :]
        h1 = h[:, 1, :] * mask[:, 1].unsqueeze(-1).float()
        pair_in = torch.cat([h0 * h1, torch.abs(h0 - h1), mean_pool], dim=-1)
        pair_feat = self.pairwise_proj(pair_in)

        z_ing = torch.cat([mean_pool, pair_feat], dim=-1)
        compat_logit = self.compat_head(z_ing)
        return z_ing, compat_logit

    def forward_policy(
        self,
        z_ing: torch.Tensor,
        goal_vec: torch.Tensor,
        chosen_op1: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        g_emb = self.goal_encoder(goal_vec)
        trunk = self.policy_trunk(torch.cat([z_ing, g_emb], dim=-1))
        op1_logits = self.op1_head(trunk)
        if chosen_op1 is None:
            chosen_op1 = torch.argmax(op1_logits, dim=-1)
        op1_e = self.op1_embed(chosen_op1)
        trunk_op1 = torch.cat([trunk, op1_e], dim=-1)

        return {
            "op1_logits": op1_logits,
            "op2_logits": self.op2_head(trunk_op1),
            "temp_logits": self.temp_head(trunk_op1),
            "press_logits": self.press_head(trunk_op1),
            "atmos_logits": self.atmos_head(trunk_op1),
            "stoich_logits": self.stoich_head(trunk),
        }

    def forward_world_model(
        self,
        z_ing: torch.Tensor,
        op1_idx: torch.Tensor,
        op2_idx: torch.Tensor,
        temp_idx: torch.Tensor,
        press_idx: torch.Tensor,
        atmos_idx: torch.Tensor,
        stoich_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r_vec = torch.cat(
            [
                F.one_hot(op1_idx, NUM_OPERATIONS).float(),
                F.one_hot(op2_idx, len(STEP2_OPERATIONS)).float(),
                F.one_hot(temp_idx, NUM_TEMP_REGIMES).float(),
                F.one_hot(press_idx, NUM_PRESSURE_REGIMES).float(),
                F.one_hot(atmos_idx, NUM_ATMOSPHERES).float(),
                F.one_hot(stoich_idx, NUM_STOICH).float(),
            ],
            dim=-1,
        )
        z_rec = self.recipe_encoder(r_vec)
        w_feat = self.world_trunk(torch.cat([z_ing, z_rec], dim=-1))

        validity_logit = self.validity_head(w_feat).squeeze(-1)
        physics_pred = torch.sigmoid(self.physics_head(w_feat).squeeze(-1))
        utility_prob = torch.sigmoid(self.utility_head(w_feat).squeeze(-1))
        prop_pred = torch.sigmoid(self.property_head(w_feat))
        cap_logits = self.capability_head(w_feat)
        reward_pred = self.reward_head(w_feat).squeeze(-1)

        return {
            "validity_logit": validity_logit,
            "validity_prob": torch.sigmoid(validity_logit),
            "physics_pred": physics_pred,
            "utility_prob": utility_prob,
            "prop_pred": prop_pred,
            "cap_logits": cap_logits,
            "cap_probs": torch.sigmoid(cap_logits),
            "reward_pred": reward_pred,
        }


class SharedBrain:
    """Thread-safe GPU neural manager shared across all parallel Inventor workers."""

    def __init__(
        self,
        checkpoint_path: str = "data/inventor_net.pt",
        lr: float = 1.5e-3,
        device: Optional[str] = None,
    ):
        self.lock = threading.RLock()
        self.checkpoint_path = checkpoint_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = InventorNet(hidden_dim=160).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )

        self.replay_buffer: List[Dict[str, Any]] = []
        self.train_steps: int = 0
        self.loss_history: List[Dict[str, float]] = []
        self.pair_compatibility_memory: Dict[Tuple[str, str], List[float]] = {}

        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def save_checkpoint(self) -> None:
        with self.lock:
            os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "train_steps": self.train_steps,
                    "loss_history": self.loss_history[-200:],
                },
                self.checkpoint_path,
            )

    def load_checkpoint(self) -> None:
        try:
            with self.lock:
                ckpt = torch.load(
                    self.checkpoint_path, map_location=self.device, weights_only=False
                )
                missing, unexpected = self.model.load_state_dict(
                    ckpt["model_state"], strict=False
                )
                if not missing and not unexpected:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                self.train_steps = int(ckpt.get("train_steps", 0))
                self.loss_history = list(ckpt.get("loss_history", []))
        except Exception:
            pass

    def warmstart_from_library(
        self,
        experiences: List[Any],
        items_by_id: Dict[str, LibraryItem],
    ) -> int:
        """Warm-start the experience replay buffer and pairwise compatibility memory from saved library history."""
        added = 0
        with self.lock:
            if self.replay_buffer:
                return 0
            for exp in experiences:
                bp = exp.blueprint
                verdict = exp.verdict
                used_items = [
                    items_by_id[ing.item_id]
                    for ing in bp.ingredients
                    if ing.item_id in items_by_id
                ]
                if not used_items:
                    continue
                self.record_experience(
                    blueprint=bp,
                    items_used=used_items,
                    verdict_valid=verdict.valid,
                    physics_score=verdict.physics_score,
                    true_properties=verdict.properties,
                    true_capabilities=verdict.capabilities,
                    reward=exp.reward,
                    utility_score=getattr(verdict, "utility_score", 0.85 if verdict.valid else 0.1),
                )
                added += 1
        return added

    def _compute_op1_prior(
        self, items: List[LibraryItem], goal_caps: Optional[List[str]] = None
    ) -> np.ndarray:
        """Compute soft prior bias over Step 1 operation given ingredient properties/capabilities and goal."""
        op1_bias = np.full(NUM_OPERATIONS, -1.5, dtype=np.float32)
        tiers = [it.tier for it in items]
        phases = {it.phase for it in items}
        caps = {c for it in items for c in it.capabilities}
        g_caps = set(goal_caps or [])
        names_lower = " ".join(it.name.lower() for it in items)
        max_tier = max(tiers)
        min_tier = min(tiers)

        if min_tier == 0:
            # Subatomic only works with nucleosynthesis or ionize_bind
            op1_bias[OP_TO_IDX["nucleosynthesis_fuse"]] = 4.5
            op1_bias[OP_TO_IDX["ionize_bind"]] = 4.5
            return op1_bias

        item_ids = {it.item_id for it in items}
        has_pure_element_dopant = (
            len(items) == 2
            and any(it.item_id in ("elem_al", "elem_li") for it in items)
            and any(it.item_id == "elem_si" or "wafer" in it.name.lower() for it in items)
        )
        all_oxides = all(
            "oxide" in it.name.lower() or "dioxide" in it.name.lower() or "aluminate" in it.name.lower()
            for it in items
        )
        has_true_reductant = any(
            it.item_id in ("elem_c", "elem_h", "elem_al", "elem_ca", "elem_si")
            for it in items
        )
        has_ceramic_phase = any(
            it.category in ("ceramic", "glass")
            or "oxide" in it.name.lower()
            or "carbide" in it.name.lower()
            or "aluminate" in it.name.lower()
            or "silicate" in it.name.lower()
            for it in items
        )
        has_metal_or_intermetallic = any(
            it.category == "alloy"
            or "conductor" in it.capabilities
            or "steel" in it.name.lower()
            or "aluminide" in it.name.lower()
            or "silicide" in it.name.lower()
            for it in items
        )
        has_copper_wire = any(
            it.item_id == "elem_cu" or "copper" in it.name.lower() or "bronze" in it.name.lower()
            for it in items
        )

        # Goal-directed engineering operations when building components/devices
        if g_caps & {"energy_storage", "actuator", "heating_element", "sensor"}:
            op1_bias.fill(-12.0)  # Strictly restrict to engineering/fabrication ops
            if any(it.category == "component" for it in items):
                op1_bias[OP_TO_IDX["wire_circuit"]] = 7.5
                op1_bias[OP_TO_IDX["assemble_mechanically"]] = 7.0
            elif "electrolyte" in caps and ("water" in names_lower or "aqueous" in phases or "liquid" in phases):
                op1_bias[OP_TO_IDX["seal_encapsulate"]] = 7.0
            elif has_copper_wire and ("iron" in names_lower or "steel" in names_lower or "silicide" in names_lower or "disilicide" in names_lower):
                op1_bias[OP_TO_IDX["wire_circuit"]] = 7.5
            elif has_ceramic_phase and has_metal_or_intermetallic:
                op1_bias[OP_TO_IDX["sinter_fire"]] = 7.0
            elif has_pure_element_dopant:
                op1_bias[OP_TO_IDX["dope_diffuse"]] = 6.8
            elif "semiconductor" in caps:
                # Semiconductor heterojunctions / thermoelectric junctions require high-T diffusion bonding (sinter_fire)
                op1_bias[OP_TO_IDX["sinter_fire"]] = 7.0
                if has_copper_wire:
                    op1_bias[OP_TO_IDX["wire_circuit"]] = 6.0
            else:
                op1_bias[OP_TO_IDX["sinter_fire"]] = 6.0
                if has_copper_wire:
                    op1_bias[OP_TO_IDX["wire_circuit"]] = 6.0
            return op1_bias

        if "optical" in g_caps and ("oxide" in names_lower or "dioxide" in names_lower):
            op1_bias.fill(-12.0)
            op1_bias[OP_TO_IDX["alloy_melt"]] = 6.5
            op1_bias[OP_TO_IDX["sinter_fire"]] = 5.5
            return op1_bias

        if len(items) == 1:
            # Single-item transformation: shaping, drawing wire, sintering, calcining
            it = items[0]
            op1_bias.fill(-10.0)
            if "conductor" in it.capabilities or it.properties.get("malleability", 0) > 0.6:
                op1_bias[OP_TO_IDX["draw_extrude"]] = 6.0
                op1_bias[OP_TO_IDX["forge_shape"]] = 5.0
            if "refractory" in it.capabilities or "oxide" in it.name.lower() or "carbide" in it.name.lower():
                op1_bias[OP_TO_IDX["sinter_fire"]] = 6.0
            if "aqueous" in it.phase or "liquid" in it.phase or "water" in it.name.lower():
                op1_bias[OP_TO_IDX["electrolyze"]] = 6.0
            return op1_bias

        if max_tier <= 2:
            if "water" in names_lower:
                op1_bias.fill(-10.0)
                op1_bias[OP_TO_IDX["dissolve"]] = 7.0
                op1_bias[OP_TO_IDX["mix"]] = 6.5
                return op1_bias

            if all_oxides:
                op1_bias.fill(-10.0)
                op1_bias[OP_TO_IDX["alloy_melt"]] = 6.5
                op1_bias[OP_TO_IDX["sinter_fire"]] = 6.0
                return op1_bias

            if "oxide" in names_lower and has_true_reductant:
                op1_bias.fill(-10.0)
                op1_bias[OP_TO_IDX["smelt_reduce"]] = 6.5
                return op1_bias

            if "elem_o" in item_ids or "elem_cl" in item_ids:
                op1_bias[OP_TO_IDX["combust_burn"]] = 6.0
                op1_bias[OP_TO_IDX["roast_oxidize"]] = 4.5

            if "elem_h" in item_ids and "elem_o" not in item_ids and "elem_cl" not in item_ids:
                op1_bias.fill(-10.0)
                op1_bias[OP_TO_IDX["heat"]] = 5.5
                op1_bias[OP_TO_IDX["catalytic_synthesize"]] = 5.5
                return op1_bias

            if "elem_n" in item_ids and "elem_si" in item_ids:
                op1_bias.fill(-10.0)
                op1_bias[OP_TO_IDX["sinter_fire"]] = 6.5
                op1_bias[OP_TO_IDX["heat"]] = 5.5
                return op1_bias

            if "water" in names_lower or "aqueous" in phases or "liquid" in phases:
                op1_bias[OP_TO_IDX["dissolve"]] = 5.0
                op1_bias[OP_TO_IDX["mix"]] = 4.5
                op1_bias[OP_TO_IDX["boil_reflux"]] = 3.8
                op1_bias[OP_TO_IDX["electrolyze"]] = 4.0

            if "acid" in caps and "base" in caps:
                op1_bias[OP_TO_IDX["mix"]] = 5.0
                op1_bias[OP_TO_IDX["crystallize_precipitate"]] = 4.5
                op1_bias[OP_TO_IDX["dissolve"]] = 4.0

            if all(it.phase == "solid" for it in items):
                if "oxide" in names_lower or "silica" in names_lower:
                    op1_bias[OP_TO_IDX["sinter_fire"]] = 5.5
                    op1_bias[OP_TO_IDX["alloy_melt"]] = 5.0
                else:
                    op1_bias[OP_TO_IDX["alloy_melt"]] = 5.5
                    op1_bias[OP_TO_IDX["sinter_fire"]] = 4.5

            if has_pure_element_dopant:
                op1_bias[OP_TO_IDX["dope_diffuse"]] = 5.8
                op1_bias[OP_TO_IDX["alloy_melt"]] = 4.8

        if max_tier >= 3:
            op1_bias[OP_TO_IDX["dope_diffuse"]] = -10.0  # Disable dope_diffuse unless pure element dopant
            if has_pure_element_dopant:
                op1_bias[OP_TO_IDX["dope_diffuse"]] = 6.0
            if max_tier >= 4 and any(it.category == "component" for it in items):
                # Tier 5 Mechanism / Device Engineering from Tier 4 components
                op1_bias.fill(-10.0)
                op1_bias[OP_TO_IDX["wire_circuit"]] = 7.2
                op1_bias[OP_TO_IDX["assemble_mechanically"]] = 6.8
                return op1_bias
            if has_copper_wire and ("iron" in names_lower or "steel" in names_lower or "silicide" in names_lower or "disilicide" in names_lower):
                op1_bias[OP_TO_IDX["wire_circuit"]] = 6.8
            elif "electrolyte" in caps and ("water" in names_lower or "aqueous" in phases or "liquid" in phases):
                op1_bias[OP_TO_IDX["seal_encapsulate"]] = 6.0
            elif has_ceramic_phase and has_metal_or_intermetallic:
                # High-temp solid-state/liquid-phase cermet sintering
                op1_bias[OP_TO_IDX["alloy_melt"]] = -8.0
                op1_bias[OP_TO_IDX["sinter_fire"]] = 7.0
            elif "semiconductor" in caps:
                op1_bias[OP_TO_IDX["sinter_fire"]] = 6.5
                if has_copper_wire:
                    op1_bias[OP_TO_IDX["wire_circuit"]] = 6.0
            else:
                op1_bias[OP_TO_IDX["sinter_fire"]] = 5.5
                op1_bias[OP_TO_IDX["alloy_melt"]] = 5.0
                if has_copper_wire:
                    op1_bias[OP_TO_IDX["wire_circuit"]] = 5.2

        return op1_bias

    def _compute_conditional_step_biases(
        self, op1_idx_val: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Given Step 1 operation, return physical envelope biases for (op2, temp, press, atmos)."""
        op1_name = PROCESS_OPERATIONS[op1_idx_val]
        op2_b = np.full(len(STEP2_OPERATIONS), -3.0, dtype=np.float32)
        op2_b[0] = 3.5  # Default strongly to 1-step recipe ("none")

        temp_b = np.full(NUM_TEMP_REGIMES, -2.5, dtype=np.float32)
        press_b = np.full(NUM_PRESSURE_REGIMES, -2.0, dtype=np.float32)
        atmos_b = np.full(NUM_ATMOSPHERES, -2.0, dtype=np.float32)

        press_b[1] = 3.0  # Default 1.0 atm

        if op1_name in ("nucleosynthesis_fuse", "ionize_bind"):
            temp_b[6] = 4.5  # 5000C
            press_b[2] = 3.0  # 50 atm
            press_b[3] = 3.5  # 500 atm
            atmos_b[ATMOS_TO_IDX["vacuum"]] = 4.5

        elif op1_name == "smelt_reduce":
            temp_b[5] = 6.0  # 1500C
            atmos_b[ATMOS_TO_IDX["reducing_h2_co"]] = 5.0
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 3.0
            op2_b[STEP2_TO_IDX["forge_shape"]] = 1.2
            op2_b[STEP2_TO_IDX["quench_temper"]] = 1.2

        elif op1_name == "alloy_melt":
            temp_b[5] = 5.5  # 1500C
            temp_b[4] = 2.5  # 850C
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 5.0
            op2_b[STEP2_TO_IDX["quench_temper"]] = 1.8
            op2_b[STEP2_TO_IDX["draw_extrude"]] = 1.5
            op2_b[STEP2_TO_IDX["forge_shape"]] = 1.5

        elif op1_name in ("calcine", "sinter_fire"):
            temp_b[5] = 5.5  # 1500C
            temp_b[4] = 3.5  # 850C
            atmos_b[ATMOS_TO_IDX["oxidizing_air"]] = 3.5
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 3.5

        elif op1_name in ("combust_burn", "roast_oxidize"):
            temp_b[3] = 4.0  # 350C
            temp_b[4] = 4.0  # 850C
            atmos_b[ATMOS_TO_IDX["oxidizing_air"]] = 4.5

        elif op1_name in ("catalytic_synthesize", "polymerize", "heat"):
            temp_b[3] = 4.5  # 350C
            temp_b[4] = 2.5  # 850C
            press_b[2] = 3.2  # 50 atm
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 4.5
            atmos_b[ATMOS_TO_IDX["reducing_h2_co"]] = 3.5

        elif op1_name == "dope_diffuse":
            temp_b[5] = 4.5  # 1500C
            temp_b[4] = 4.5  # 850C
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 5.0
            atmos_b[ATMOS_TO_IDX["vacuum"]] = 3.5

        elif op1_name in ("dissolve", "mix", "crystallize_precipitate", "electrolyze"):
            temp_b[1] = 4.5  # 25C
            temp_b[2] = 2.5  # 100C
            atmos_b[ATMOS_TO_IDX["aqueous_solution"]] = 4.5
            if op1_name == "dissolve":
                op2_b[STEP2_TO_IDX["crystallize_precipitate"]] = 1.8
                op2_b[STEP2_TO_IDX["electrolyze"]] = 2.0
                op2_b[STEP2_TO_IDX["boil_reflux"]] = 1.5

        elif op1_name in ("boil_reflux", "distill_fractionate"):
            temp_b[2] = 4.5  # 100C
            atmos_b[ATMOS_TO_IDX["aqueous_solution"]] = 4.0
            atmos_b[ATMOS_TO_IDX["oxidizing_air"]] = 2.5

        elif op1_name in ("draw_extrude", "forge_shape", "quench_temper", "coat_plate"):
            temp_b[3] = 4.0  # 350C
            temp_b[4] = 3.5  # 850C
            temp_b[1] = 2.5  # 25C
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 4.5

        elif op1_name in ("assemble_mechanically", "wire_circuit", "seal_encapsulate"):
            temp_b.fill(-12.0)
            temp_b[1] = 6.0  # 25C
            atmos_b.fill(-6.0)
            atmos_b[ATMOS_TO_IDX["inert_argon"]] = 6.0
            if op1_name == "assemble_mechanically":
                op2_b[STEP2_TO_IDX["wire_circuit"]] = 2.2
                op2_b[STEP2_TO_IDX["seal_encapsulate"]] = 2.2

        return op2_b, temp_b, press_b, atmos_b

    def _assign_ingredient_roles(
        self, items: List[LibraryItem], op1_name: str
    ) -> List[str]:
        roles: List[str] = []
        for idx, it in enumerate(items):
            caps = set(it.capabilities)
            if idx == 0:
                if op1_name in ("assemble_mechanically", "wire_circuit", "seal_encapsulate"):
                    roles.append(
                        "functional_core"
                        if ("conductor" in caps or "semiconductor" in caps or "magnetic" in caps)
                        else "structural_part"
                    )
                else:
                    roles.append("primary_reactant")
            else:
                if "catalyst" in caps and op1_name == "catalytic_synthesize":
                    roles.append("catalyst")
                elif ("solvent" in caps or "water" in it.name.lower()) and op1_name in (
                    "dissolve",
                    "mix",
                    "boil_reflux",
                    "electrolyze",
                    "crystallize_precipitate",
                ):
                    roles.append("solvent_medium")
                elif "reducing_agent" in caps and op1_name == "smelt_reduce":
                    roles.append("reducing_agent")
                elif "oxidizer" in caps and op1_name in ("combust_burn", "roast_oxidize"):
                    roles.append("oxidizer")
                elif op1_name == "dope_diffuse":
                    roles.append("dopant")
                elif op1_name in ("assemble_mechanically", "wire_circuit", "seal_encapsulate"):
                    roles.append(
                        "structural_part"
                        if ("structural" in caps or "vessel" in caps or "insulator" in caps)
                        else "functional_core"
                    )
                else:
                    roles.append("secondary_reactant")
        return roles

    def _apply_equipment_gate_masks(
        self,
        op1_logits: torch.Tensor,
        op2_logits: Optional[torch.Tensor],
        temp_logits: Optional[torch.Tensor],
        press_logits: Optional[torch.Tensor],
        atmos_logits: Optional[torch.Tensor],
        equipment_gates: Optional[Dict[str, Dict[str, Any]]],
    ) -> None:
        """In-place mask out operations and regimes whose technological equipment gates are still locked."""
        if not equipment_gates:
            return
        NEG_INF = -25.0

        def is_locked(gate: str) -> bool:
            return not bool(equipment_gates.get(gate, {}).get("unlocked", True))

        if is_locked("wet_chemistry"):
            for op in ("dissolve", "boil_reflux", "distill_fractionate", "crystallize_precipitate"):
                op1_logits[:, OP_TO_IDX[op]] += NEG_INF
                if op2_logits is not None:
                    op2_logits[:, STEP2_TO_IDX[op]] += NEG_INF
            if atmos_logits is not None:
                atmos_logits[:, ATMOS_TO_IDX["aqueous_solution"]] += NEG_INF

        if is_locked("high_temp_1500c"):
            op1_logits[:, OP_TO_IDX["smelt_reduce"]] += NEG_INF
            if op2_logits is not None:
                op2_logits[:, STEP2_TO_IDX["smelt_reduce"]] += NEG_INF
            if temp_logits is not None:
                temp_logits[:, 5] += NEG_INF  # 1500C locked until refractory oxide/crucible invented

        if is_locked("electrochemistry"):
            for op in ("electrolyze", "coat_plate"):
                op1_logits[:, OP_TO_IDX[op]] += NEG_INF
                if op2_logits is not None:
                    op2_logits[:, STEP2_TO_IDX[op]] += NEG_INF

        if is_locked("high_pressure_50atm") and press_logits is not None:
            press_logits[:, 2] += NEG_INF  # 50 atm
            press_logits[:, 3] += NEG_INF  # 500 atm

        if is_locked("precision_fabrication"):
            for op in ("wire_circuit", "dope_diffuse", "seal_encapsulate", "polymerize"):
                op1_logits[:, OP_TO_IDX[op]] += NEG_INF
                if op2_logits is not None:
                    op2_logits[:, STEP2_TO_IDX[op]] += NEG_INF

        if is_locked("plasma_arc_5000c"):
            for op in ("nucleosynthesis_fuse", "ionize_bind"):
                op1_logits[:, OP_TO_IDX[op]] += NEG_INF
                if op2_logits is not None:
                    op2_logits[:, STEP2_TO_IDX[op]] += NEG_INF
            if temp_logits is not None:
                temp_logits[:, 6] += NEG_INF  # 5000C locked until Tier 4 magnetic/power component invented

    def _resolve_equipment_used(
        self,
        op1_name: str,
        op2_name: str,
        temp_c: float,
        press_atm: float,
        atmos_name: str,
        equipment_gates: Optional[Dict[str, Dict[str, Any]]],
        excitation_field: str = "none",
    ) -> List[str]:
        """Determine which unlocked lab/industrial equipment items are utilized by this recipe."""
        if not equipment_gates:
            return []
        used: List[str] = []
        ops = {op1_name, op2_name}

        apparatus_labels = {
            "wet_chemistry": "Aqueous Reflux & Crystallization Reactor",
            "high_temp_1500c": "Refractory Alumina Crucible & High-Temp Induction Furnace",
            "electrochemistry": "Electrolytic Cell & Molten/Aqueous Salt Bridge",
            "high_pressure_50atm": "High-Pressure Autoclave Synthesis Reactor",
            "precision_fabrication": "Precision Diffusion Furnace & Micro-Assembly Rig",
            "plasma_arc_5000c": "Magnetic Plasma Arc & Fusion Chamber",
        }

        def add_eq(gate: str) -> None:
            eq = equipment_gates.get(gate, {}).get("equipment")
            if eq:
                label = f"{apparatus_labels.get(gate, gate)} [built with {eq}]"
                if label not in used:
                    used.append(label)

        if (
            ops & {"dissolve", "boil_reflux", "distill_fractionate", "crystallize_precipitate"}
            or atmos_name == "aqueous_solution"
        ):
            add_eq("wet_chemistry")
        if temp_c >= 1100.0 or "smelt_reduce" in ops or "alloy_melt" in ops:
            add_eq("high_temp_1500c")
        if ops & {"electrolyze", "coat_plate"} or excitation_field == "dc_voltage_bias":
            add_eq("electrochemistry")
        if press_atm >= 10.0:
            add_eq("high_pressure_50atm")
        if ops & {"wire_circuit", "dope_diffuse", "seal_encapsulate", "polymerize"} or excitation_field != "none":
            add_eq("precision_fabrication")
        if temp_c >= 3000.0 or ops & {"nucleosynthesis_fuse", "ionize_bind"} or excitation_field in ("ionizing_radiation_beam", "high_voltage_discharge"):
            add_eq("plasma_arc_5000c")

        return used

    def generate_and_rank_blueprints(
        self,
        library_items: List[LibraryItem],
        proposer_id: str,
        strategy: str,
        target_goal_desc: str,
        goal_props: Optional[Dict[str, float]] = None,
        goal_caps: Optional[List[str]] = None,
        seen_blueprint_ids: Optional[Set[str]] = None,
        candidate_item_subsets: Optional[List[List[LibraryItem]]] = None,
        equipment_gates: Optional[Dict[str, Dict[str, Any]]] = None,
        num_candidates: int = 96,
        top_k: int = 3,
        exploration_temp: float = 0.75,
    ) -> List[Blueprint]:
        """Sample candidate ingredient combinations, run batched GPU policy + world model, and return top unseen Blueprints."""
        if not library_items:
            return []
        seen = seen_blueprint_ids or set()

        g_vec = np.full(GOAL_DIM, 0.5, dtype=np.float32)
        if goal_props:
            for i, k in enumerate(PROPERTY_KEYS):
                if k in goal_props:
                    g_vec[i] = float(goal_props[k])
        if goal_caps:
            for c in goal_caps:
                if c in CAPABILITY_TO_IDX:
                    g_vec[NUM_PROPERTIES + CAPABILITY_TO_IDX[c]] = 1.0

        if candidate_item_subsets is None:
            candidate_item_subsets = []
            for _ in range(num_candidates):
                r = random.random()
                if r < 0.12:
                    k = 1
                elif r < 0.80 or len(library_items) < 3:
                    k = min(2, len(library_items))
                else:
                    k = min(3, len(library_items))
                candidate_item_subsets.append(random.sample(library_items, k))

        B = len(candidate_item_subsets)
        feat_tensor = np.zeros((B, MAX_INGREDIENTS, ITEM_FEAT_DIM), dtype=np.float32)
        mask_tensor = np.zeros((B, MAX_INGREDIENTS), dtype=np.bool_)
        op1_priors = np.zeros((B, NUM_OPERATIONS), dtype=np.float32)

        for b, subset in enumerate(candidate_item_subsets):
            for j, it in enumerate(subset[:MAX_INGREDIENTS]):
                feat_tensor[b, j] = extract_item_feature_vector(it)
                mask_tensor[b, j] = True
            op1_priors[b] = self._compute_op1_prior(subset, goal_caps=goal_caps)

        with self.lock, torch.no_grad():
            self.model.eval()
            t_feats = torch.from_numpy(feat_tensor).to(self.device)
            t_mask = torch.from_numpy(mask_tensor).to(self.device)
            t_goal = (
                torch.from_numpy(g_vec)
                .unsqueeze(0)
                .expand(B, -1)
                .to(self.device)
            )
            t_op1_prior = torch.from_numpy(op1_priors).to(self.device)

            z_ing, compat_logit = self.model.encode_ingredients(t_feats, t_mask)
            compat_prob = torch.sigmoid(compat_logit).squeeze(-1)

            pol = self.model.forward_policy(z_ing, t_goal)
            op1_logits = (pol["op1_logits"] + t_op1_prior) / max(0.25, exploration_temp)
            self._apply_equipment_gate_masks(
                op1_logits, None, None, None, None, equipment_gates
            )
            op1_idx = torch.multinomial(F.softmax(op1_logits, dim=-1), 1).squeeze(-1)
            op1_np_temp = op1_idx.cpu().numpy()

            # Compute operation-conditioned physical biases for Step 2, Temp, Pressure, Atmosphere
            op2_priors = np.zeros((B, len(STEP2_OPERATIONS)), dtype=np.float32)
            temp_priors = np.zeros((B, NUM_TEMP_REGIMES), dtype=np.float32)
            press_priors = np.zeros((B, NUM_PRESSURE_REGIMES), dtype=np.float32)
            atmos_priors = np.zeros((B, NUM_ATMOSPHERES), dtype=np.float32)

            for b in range(B):
                o2b, tb, pb, ab = self._compute_conditional_step_biases(
                    int(op1_np_temp[b])
                )
                op2_priors[b] = o2b
                temp_priors[b] = tb
                press_priors[b] = pb
                atmos_priors[b] = ab

            pol2 = self.model.forward_policy(z_ing, t_goal, chosen_op1=op1_idx)

            t_op2_p = torch.from_numpy(op2_priors).to(self.device)
            t_temp_p = torch.from_numpy(temp_priors).to(self.device)
            t_press_p = torch.from_numpy(press_priors).to(self.device)
            t_atmos_p = torch.from_numpy(atmos_priors).to(self.device)

            op2_logits = (pol2["op2_logits"] + t_op2_p) / max(0.25, exploration_temp)
            temp_logits = (pol2["temp_logits"] + t_temp_p) / max(0.25, exploration_temp)
            press_logits = (pol2["press_logits"] + t_press_p) / max(
                0.25, exploration_temp
            )
            atmos_logits = (pol2["atmos_logits"] + t_atmos_p) / max(
                0.25, exploration_temp
            )
            self._apply_equipment_gate_masks(
                op1_logits,
                op2_logits,
                temp_logits,
                press_logits,
                atmos_logits,
                equipment_gates,
            )

            op2_idx = torch.multinomial(F.softmax(op2_logits, dim=-1), 1).squeeze(-1)
            temp_idx = torch.multinomial(F.softmax(temp_logits, dim=-1), 1).squeeze(-1)
            press_idx = torch.multinomial(
                F.softmax(press_logits, dim=-1), 1
            ).squeeze(-1)
            atmos_idx = torch.multinomial(
                F.softmax(atmos_logits, dim=-1), 1
            ).squeeze(-1)

            stoich_logits = pol2["stoich_logits"] / max(0.25, exploration_temp)
            stoich_idx = torch.multinomial(
                F.softmax(stoich_logits, dim=-1), 1
            ).squeeze(-1)

            world = self.model.forward_world_model(
                z_ing=z_ing,
                op1_idx=op1_idx,
                op2_idx=op2_idx,
                temp_idx=temp_idx,
                press_idx=press_idx,
                atmos_idx=atmos_idx,
                stoich_idx=stoich_idx,
            )

            val_probs = world["validity_prob"].cpu().numpy()
            phys_preds = world["physics_pred"].cpu().numpy()
            util_probs = world["utility_prob"].cpu().numpy()
            prop_preds = world["prop_pred"].cpu().numpy()
            cap_probs = world["cap_probs"].cpu().numpy()
            rew_preds = world["reward_pred"].cpu().numpy()
            compat_np = compat_prob.cpu().numpy()

            op1_np = op1_idx.cpu().numpy()
            op2_np = op2_idx.cpu().numpy()
            temp_np = temp_idx.cpu().numpy()
            press_np = press_idx.cpu().numpy()
            atmos_np = atmos_idx.cpu().numpy()
            stoich_np = stoich_idx.cpu().numpy()

        scored_blueprints: List[Tuple[float, Blueprint]] = []
        local_seen: Set[str] = set()
        op_counts: Dict[int, int] = {}
        with self.lock:
            for rec in self.replay_buffer:
                o_i = int(rec.get("op1", 0))
                op_counts[o_i] = op_counts.get(o_i, 0) + 1

        existing_parent_sets: Dict[Tuple[str, ...], int] = {}
        for it in library_items:
            if it.parent_ids:
                pkey = tuple(sorted(it.parent_ids))
                existing_parent_sets[pkey] = existing_parent_sets.get(pkey, 0) + 1

        for b in range(B):
            subset = list(candidate_item_subsets[b])
            op1_name = PROCESS_OPERATIONS[int(op1_np[b])]
            op2_name = STEP2_OPERATIONS[int(op2_np[b])]
            _, temp_c, _ = TEMP_REGIMES[int(temp_np[b])]
            _, press_atm, _ = PRESSURE_REGIMES[int(press_np[b])]
            atmos_name = ATMOSPHERE_REGIMES[int(atmos_np[b])]
            _, raw_ratios = STOICHIOMETRY_TEMPLATES[int(stoich_np[b])]
            ratios = list(raw_ratios)

            sub_id_set = {it.item_id for it in subset}
            sub_names_low = " ".join(it.name.lower() for it in subset)
            custom_roles: Optional[List[str]] = None
            excitation_field: str = "none"
            custom_details: Optional[str] = None

            # Enforce host-rich stoichiometry for doping, gases, nitrides, wafers, aqueous chemistry, and functional devices
            if all(it.tier == 0 for it in subset):
                ratios = [1 for _ in subset]
                op1_name = "ionize_bind"
                temp_c = 5000.0
                press_atm = 1.0
                atmos_name = "vacuum"
                excitation_field = "high_voltage_discharge"
            elif "sub_neutron" in sub_id_set and any(it.item_id == "elem_si" or "silicon" in it.name.lower() for it in subset):
                # Neutron Transmutation Doping (NTD): 30Si(n,γ)31Si -> 31P + β- followed by 850°C argon anneal
                n_it = next(it for it in subset if it.item_id == "sub_neutron")
                si_it = next(it for it in subset if it.item_id != "sub_neutron")
                subset = [si_it, n_it]
                ratios = [10, 1]
                op1_name = "dope_diffuse"
                op2_name = "heat"
                temp_c = 850.0
                press_atm = 1.0
                atmos_name = "inert_argon"
                excitation_field = "ionizing_radiation_beam"
                custom_roles = ["active_medium", "excitation_source"]
                custom_details = "Thermal neutron flux capture on 30Si (30Si + n0 -> 31Si -> 31P + β-) with 850°C argon lattice-annealing to activate substitutional phosphorus n-type donors"
            elif "sub_photon" in sub_id_set and any(it.tier >= 1 for it in subset):
                # Photonic / Light Excitation (Photovoltaic Solar Cell, Photodiode, or Photoelectrochemical Water Splitting)
                ph_it = next(it for it in subset if it.item_id == "sub_photon")
                others = [it for it in subset if it.item_id != "sub_photon"]
                water_it = next((it for it in others if "water" in it.name.lower()), None)
                if water_it and len(others) == 2:
                    # Photoelectrochemical Solar Water-Splitting Cell (Semiconductor/Oxide Photoelectrode + Water + Photon hν)
                    photo_el = next(it for it in others if it.item_id != water_it.item_id)
                    subset = [photo_el, water_it, ph_it]
                    ratios = [1, 2, 1]
                    op1_name = "electrolyze"
                    op2_name = "none"
                    temp_c = 25.0
                    press_atm = 1.0
                    atmos_name = "aqueous_solution"
                    excitation_field = "uv_solar_light"
                    custom_roles = ["active_medium", "solvent_medium", "excitation_source"]
                    custom_details = "Solar photonic illumination (hν >= E_g) exciting electron-hole pairs on photoelectrode in water to drive photoelectrochemical water splitting (2H2O -> 2H2 + O2)"
                else:
                    # Solid-State Photovoltaic Solar Cell / Optical Photodiode (Semiconductor Junction/Wafer + Metal Contact + Photon hν)
                    semi_it = next((it for it in others if "semiconductor" in it.capabilities or "heterojunction" in it.name.lower() or "wafer" in it.name.lower() or "diode" in it.name.lower() or "carbide" in it.name.lower()), others[0])
                    rest = [it for it in others if it.item_id != semi_it.item_id]
                    subset = [semi_it] + rest + [ph_it]
                    ratios = [1 for _ in subset]
                    if any("phosphorus doped" in it.name.lower() or "wafer" in it.name.lower() for it in rest):
                        op1_name = "dope_diffuse"
                        op2_name = "wire_circuit"
                        temp_c = 850.0
                    else:
                        op1_name = "wire_circuit"
                        op2_name = "none"
                        temp_c = 25.0
                    press_atm = 1.0
                    atmos_name = "inert_argon"
                    excitation_field = "uv_solar_light"
                    custom_roles = ["active_medium"] + ["functional_core" for _ in rest] + ["excitation_source"]
                    custom_details = "Continuous UV/visible photonic illumination (hν >= E_g) across p-n / heterojunction generating separated electron-hole pairs and DC photovoltaic power"
            elif "sub_voltage" in sub_id_set and any(it.tier >= 1 for it in subset):
                # Electrical Voltage / Field / Charge Excitation (Anodization, Electroluminescent LED, Magnetic Poling, or Thermionic Emission)
                v_it = next(it for it in subset if it.item_id == "sub_voltage")
                others = [it for it in subset if it.item_id != "sub_voltage"]
                water_it = next((it for it in others if "water" in it.name.lower()), None)
                al_it = next((it for it in others if it.item_id == "elem_al"), None)
                solenoid_it = next((it for it in others if "solenoid" in it.name.lower()), None)
                elec_it = next((it for it in others if it.item_id == "sub_electron"), None)
                if al_it and water_it:
                    # Anodic Oxidation (Anodized Aluminum Dielectric Substrate)
                    subset = [al_it, water_it, v_it]
                    ratios = [2, 3, 1]
                    op1_name = "electrolyze"
                    op2_name = "coat_plate"
                    temp_c = 25.0
                    press_atm = 1.0
                    atmos_name = "aqueous_solution"
                    excitation_field = "dc_voltage_bias"
                    custom_roles = ["active_medium", "solvent_medium", "excitation_source"]
                    custom_details = "Electrochemical anodic DC voltage bias (2Al + 3H2O -> Al2O3 + 6H+ + 6e-) growing an adherent, insulating barrier Al2O3 dielectric layer on aluminum"
                elif solenoid_it and len(others) == 2:
                    # Magnetic Field Poling -> Permanent Magnet
                    mag_it = next(it for it in others if it.item_id != solenoid_it.item_id)
                    subset = [mag_it, solenoid_it, v_it]
                    ratios = [1, 1, 1]
                    op1_name = "quench_temper"
                    op2_name = "wire_circuit"
                    temp_c = 350.0
                    press_atm = 1.0
                    atmos_name = "inert_argon"
                    excitation_field = "magnetic_induction_field"
                    custom_roles = ["active_medium", "functional_core", "excitation_source"]
                    custom_details = "Energizing Electromagnet Solenoid with DC potential to impose a strong external B-field during thermal treatment, aligning ferromagnetic domains into a permanent magnet"
                elif elec_it and len(others) == 2:
                    # Thermionic Vacuum Electron Emitter / Cathode Ray Source
                    emitter_it = next(it for it in others if it.item_id != "sub_electron")
                    subset = [emitter_it, elec_it, v_it]
                    ratios = [1, 1, 1]
                    op1_name = "wire_circuit"
                    op2_name = "seal_encapsulate"
                    temp_c = 850.0
                    press_atm = 0.01
                    atmos_name = "vacuum"
                    excitation_field = "high_voltage_discharge"
                    custom_roles = ["active_medium", "excitation_source", "excitation_source"]
                    custom_details = "High-voltage electrostatic extraction field and thermal heating in vacuum (0.01 atm) overcoming cathode work function to emit a focused electron beam"
                else:
                    # Electroluminescent LED / Voltage-Biased Semiconductor Component (e.g. Silicon Carbide LED)
                    semi_it = next((it for it in others if "carbide" in it.name.lower() or "semiconductor" in it.capabilities or "heterojunction" in it.name.lower()), others[0])
                    rest = [it for it in others if it.item_id != semi_it.item_id]
                    subset = [semi_it] + rest + [v_it]
                    ratios = [1 for _ in subset]
                    op1_name = "wire_circuit"
                    op2_name = "none"
                    temp_c = 25.0
                    press_atm = 1.0
                    atmos_name = "inert_argon"
                    excitation_field = "dc_voltage_bias"
                    custom_roles = ["active_medium"] + ["functional_core" for _ in rest] + ["excitation_source"]
                    custom_details = "Forward DC voltage bias injecting electrons and holes across wide-bandgap semiconductor junction to induce radiative recombination electroluminescence (LED)"
            elif len(subset) == 2 and sub_id_set == {"elem_h", "elem_o"}:
                if subset[0].item_id == "elem_o":
                    subset = [subset[1], subset[0]]
                ratios[0], ratios[1] = 2, 1
                op1_name = "combust_burn"
                temp_c = 350.0
                press_atm = 1.0
                atmos_name = "oxidizing_air"
            elif {"elem_h", "elem_n"} <= sub_id_set and len(subset) in (2, 3):
                # Haber-Bosch Ammonia (NH3) synthesis
                h_it = next(it for it in subset if it.item_id == "elem_h")
                n_it = next(it for it in subset if it.item_id == "elem_n")
                cat_it = next((it for it in subset if it.item_id not in ("elem_h", "elem_n")), None)
                if cat_it:
                    subset = [h_it, n_it, cat_it]
                    ratios = [3, 1, 1]
                    custom_roles = ["primary_reactant", "secondary_reactant", "catalyst"]
                else:
                    subset = [h_it, n_it]
                    ratios = [3, 1]
                op1_name = "catalytic_synthesize"
                op2_name = "none"
                temp_c = 350.0
                press_atm = 50.0
                atmos_name = "inert_argon"
            elif {"elem_c", "elem_h"} <= sub_id_set and len(subset) in (2, 3) and all(it.tier <= 1 for it in subset):
                # Catalytic Methane (CH4) synthesis
                c_it = next(it for it in subset if it.item_id == "elem_c")
                h_it = next(it for it in subset if it.item_id == "elem_h")
                cat_it = next((it for it in subset if it.item_id not in ("elem_c", "elem_h")), None)
                if cat_it:
                    subset = [c_it, h_it, cat_it]
                    ratios = [1, 2, 1]
                    custom_roles = ["primary_reactant", "secondary_reactant", "catalyst"]
                else:
                    subset = [c_it, h_it]
                    ratios = [1, 2]
                op1_name = "catalytic_synthesize"
                op2_name = "none"
                temp_c = 350.0
                press_atm = 50.0
                atmos_name = "inert_argon"
            elif len(subset) == 2 and sub_id_set == {"elem_h", "elem_cl"}:
                if subset[0].item_id == "elem_cl":
                    subset = [subset[1], subset[0]]
                ratios[0], ratios[1] = 1, 1
                op1_name = "combust_burn"
                op2_name = "none"
                temp_c = 350.0
                press_atm = 1.0
                atmos_name = "inert_argon"
            elif len(subset) == 2 and sub_id_set == {"elem_s", "elem_o"}:
                if subset[0].item_id == "elem_o":
                    subset = [subset[1], subset[0]]
                ratios[0], ratios[1] = 1, 1
                op1_name = "combust_burn"
                op2_name = "none"
                temp_c = 350.0
                press_atm = 1.0
                atmos_name = "oxidizing_air"
            elif any("water" in it.name.lower() for it in subset) and not any(it.category in ("component", "mechanism") for it in subset):
                w_it = next(it for it in subset if "water" in it.name.lower())
                others = [it for it in subset if it.item_id != w_it.item_id]
                subset = others + [w_it]
                if any("cement" in it.name.lower() or "aluminate" in it.name.lower() or "hydroxide" in it.name.lower() for it in others) and len(others) >= 1:
                    # Hydraulic cement / mortar hydration
                    if len(others) == 1 and ("cement" in others[0].name.lower() or "aluminate" in others[0].name.lower()):
                        ratios = [2, 1]
                        op1_name = "mix"
                    elif len(others) == 2:
                        ratios = [1, 2, 1]
                        op1_name = "mix"
                    else:
                        ratios = [1, 1]
                        op1_name = "dissolve"
                elif len(others) == 2 and {others[0].item_id, others[1].item_id} == {"elem_al", "elem_cu"}:
                    # Aqueous Aluminum-Copper Galvanic Battery Cell
                    al_it = next(it for it in others if it.item_id == "elem_al")
                    cu_it = next(it for it in others if it.item_id == "elem_cu")
                    subset = [al_it, cu_it, w_it]
                    ratios = [1, 1, 2]
                    op1_name = "assemble_mechanically"
                    op2_name = "wire_circuit"
                    custom_roles = ["functional_core", "functional_core", "solvent_medium"]
                else:
                    ratios = [1, 1] if len(others) == 1 else [1 for _ in subset]
                    op1_name = "dissolve"
                if op1_name != "assemble_mechanically":
                    op2_name = "none"
                temp_c = 25.0
                press_atm = 1.0
                atmos_name = "aqueous_solution"
            elif ("p-type" in sub_names_low or "doped silicon wafer" in sub_names_low) and ("n-type" in sub_names_low or "phosphorus doped" in sub_names_low):
                # Silicon PN-Junction Diode / Photovoltaic Solar Cell
                ratios = [1 for _ in subset]
                op1_name = "dope_diffuse"
                op2_name = "wire_circuit"
                temp_c = 850.0
                press_atm = 1.0
                atmos_name = "inert_argon"
            elif any("wafer" in it.name.lower() for it in subset) and len(subset) == 3 and any("dioxide" in it.name.lower() or "oxide" in it.name.lower() for it in subset):
                # MOS Capacitor / Gate Stack (Doped Silicon Wafer + Oxide Dielectric + Metal Gate)
                wafer_it = next(it for it in subset if "wafer" in it.name.lower())
                ox_it = next(it for it in subset if ("dioxide" in it.name.lower() or "oxide" in it.name.lower()) and it.item_id != wafer_it.item_id)
                met_it = next(it for it in subset if it.item_id not in (wafer_it.item_id, ox_it.item_id))
                subset = [wafer_it, ox_it, met_it]
                ratios = [1, 1, 1]
                op1_name = "coat_plate"
                op2_name = "wire_circuit"
                temp_c = 350.0
                press_atm = 1.0
                atmos_name = "inert_argon"
                custom_roles = ["functional_core", "structural_part", "functional_core"]
            elif (
                len(subset) == 3
                and any(it.item_id in ("elem_li", "elem_al", "elem_ca", "elem_na") for it in subset)
                and any("chloride" in it.name.lower() or "hydroxide" in it.name.lower() or "electrolyte" in it.capabilities for it in subset)
            ):
                # Galvanic / Molten-Salt Thermal Battery Cell (Anode + Cathode + Electrolyte)
                anode_it = next(it for it in subset if it.item_id in ("elem_li", "elem_al", "elem_ca", "elem_na"))
                elec_it = next(it for it in subset if ("chloride" in it.name.lower() or "hydroxide" in it.name.lower() or "electrolyte" in it.capabilities) and it.item_id != anode_it.item_id)
                cath_it = next(it for it in subset if it.item_id not in (anode_it.item_id, elec_it.item_id))
                subset = [anode_it, cath_it, elec_it]
                ratios = [1, 1, 1]
                op1_name = "assemble_mechanically"
                op2_name = "seal_encapsulate"
                temp_c = 25.0
                atmos_name = "aqueous_solution" if "hydroxide" in elec_it.name.lower() else "inert_argon"
                press_atm = 1.0
                custom_roles = ["functional_core", "functional_core", "solvent_medium"]
            elif (
                len(subset) == 3
                and any(it.item_id == "elem_cu" for it in subset)
                and any(it.name == "Aluminum Oxide" for it in subset)
                and any("silicon" in it.name.lower() or "aluminide" in it.name.lower() or "silicide" in it.name.lower() for it in subset)
            ):
                # Bimetallic / Silicide Thermocouple Temperature Sensor
                cu_it = next(it for it in subset if it.item_id == "elem_cu")
                al2o3_it = next(it for it in subset if it.name == "Aluminum Oxide")
                leg_it = next(it for it in subset if it.item_id not in (cu_it.item_id, al2o3_it.item_id))
                subset = [cu_it, leg_it, al2o3_it]
                ratios = [1, 1, 1]
                op1_name = "wire_circuit"
                op2_name = "none"
                temp_c = 25.0
                press_atm = 1.0
                atmos_name = "inert_argon"
                custom_roles = ["functional_core", "functional_core", "structural_part"]
            elif any(it.category in ("component", "mechanism") for it in subset):
                # Assembling pre-built electrical/electromagnetic/heating components into Tier 5 Mechanisms/Devices must occur at 25°C
                comp_it = next(it for it in subset if it.category in ("component", "mechanism"))
                others = [it for it in subset if it.item_id != comp_it.item_id]
                subset = [comp_it] + others
                if "solenoid" in comp_it.name.lower() and len(subset) == 2 and ("steel" in others[0].name.lower() or "ferrosilicon" in others[0].name.lower()):
                    # Electrical Step-Up / Step-Down Transformer (2x Solenoid + 1x Silicon Steel Core)
                    ratios = [2, 1]
                    op1_name = "assemble_mechanically"
                    op2_name = "wire_circuit"
                elif "heating element" in comp_it.name.lower():
                    # High-Temperature Electric Resistance Furnace
                    ratios = [2, 3, 1][:len(subset)]
                    op1_name = "assemble_mechanically"
                    op2_name = "wire_circuit"
                else:
                    ratios = [1 for _ in subset]
                    if op1_name not in ("wire_circuit", "assemble_mechanically"):
                        op1_name = "assemble_mechanically"
                    op2_name = "wire_circuit" if op1_name == "assemble_mechanically" else "none"
                temp_c = 25.0
                press_atm = 1.0
                atmos_name = "inert_argon"
            elif any("wafer" in it.name.lower() for it in subset) and len(subset) == 2:
                # Planar silicon wafer heterojunctions must be solid-state diffusion bonded below 937°C (never melted at 1500°C!)
                if "wafer" not in subset[0].name.lower():
                    subset = [subset[1], subset[0]]
                ratios[0], ratios[1] = 1, 1
                op1_name = "sinter_fire"
                temp_c = 850.0
                press_atm = 1.0
                atmos_name = "inert_argon"
            elif op1_name == "dope_diffuse" and len(subset) == 2:
                if subset[0].item_id in ("elem_al", "elem_li") or (subset[0].tier == 1 and subset[1].tier > 1):
                    subset = [subset[1], subset[0]]
                ratios[0], ratios[1] = 3, 1
            elif len(subset) == 2 and sub_id_set == {"elem_si", "elem_n"}:
                if subset[0].item_id == "elem_n":
                    subset = [subset[1], subset[0]]
                ratios[0], ratios[1] = 3, 2
            elif op1_name == "sinter_fire" and len(subset) == 2:
                is_cer_0 = subset[0].category in ("ceramic", "glass") or "carbide" in subset[0].name.lower() or "oxide" in subset[0].name.lower()
                is_cer_1 = subset[1].category in ("ceramic", "glass") or "carbide" in subset[1].name.lower() or "oxide" in subset[1].name.lower()
                if is_cer_0 and not is_cer_1:
                    subset = [subset[1], subset[0]]
                    is_cer_0, is_cer_1 = False, True
                if not is_cer_0 and is_cer_1:
                    ratios[0], ratios[1] = 2, 1
                    atmos_name = "inert_argon"
                    if temp_c >= 1500.0 and "aluminum" in subset[0].name.lower() and "iron" not in subset[0].name.lower():
                        temp_c = 850.0
                    elif temp_c < 1500.0 and ("iron" in subset[0].name.lower() or "silicide" in subset[0].name.lower() or "aluminide" in subset[0].name.lower()):
                        temp_c = 1500.0

            roles = custom_roles if custom_roles is not None else self._assign_ingredient_roles(subset, op1_name)

            ings: List[BlueprintIngredient] = []
            for j, it in enumerate(subset):
                ings.append(
                    BlueprintIngredient(
                        item_id=it.item_id,
                        item_name=it.name,
                        formula=it.formula_or_spec,
                        ratio=ratios[j] if len(subset) > 1 else 1,
                        role=roles[j],
                    )
                )

            steps: List[ProcessStep] = [
                ProcessStep(
                    step_order=1,
                    operation=op1_name,
                    temperature_c=temp_c,
                    pressure_atm=press_atm,
                    atmosphere=atmos_name,
                    duration="20 min",
                    details=custom_details or f"Primary {op1_name} stage in {atmos_name}",
                    excitation_field=excitation_field,
                )
            ]
            if op2_name != "none" and op2_name != op1_name:
                step2_temp = (
                    25.0
                    if op2_name
                    in (
                        "quench_temper",
                        "crystallize_precipitate",
                        "assemble_mechanically",
                        "wire_circuit",
                        "seal_encapsulate",
                    )
                    else temp_c
                )
                steps.append(
                    ProcessStep(
                        step_order=2,
                        operation=op2_name,
                        temperature_c=step2_temp,
                        pressure_atm=press_atm if atmos_name == "vacuum" else 1.0,
                        atmosphere=atmos_name,
                        duration="15 min",
                        details=f"Secondary {op2_name} stage",
                        excitation_field=excitation_field,
                    )
                )

            bp_id = Blueprint.compute_canonical_id(ings, steps)
            if bp_id in seen or bp_id in local_seen:
                continue
            local_seen.add(bp_id)

            pred_props_dict = {
                PROPERTY_KEYS[i]: round(float(prop_preds[b, i]), 4)
                for i in range(NUM_PROPERTIES)
            }
            pred_caps_list = [
                CAPABILITY_VOCAB[i]
                for i in range(NUM_CAPABILITIES)
                if float(cap_probs[b, i]) >= 0.45
            ]

            neural_pred = NeuralPrediction(
                predicted_validity=round(float(val_probs[b]), 4),
                predicted_compatibility=round(float(compat_np[b]), 4),
                predicted_physics_score=round(float(phys_preds[b]), 4),
                predicted_utility=round(float(util_probs[b]), 4),
                predicted_properties=pred_props_dict,
                predicted_capabilities=pred_caps_list,
                expected_reward=round(float(rew_preds[b]), 4),
                policy_indices={
                    "op1": int(op1_np[b]),
                    "op2": int(op2_np[b]),
                    "temp": int(temp_np[b]),
                    "press": int(press_np[b]),
                    "atmos": int(atmos_np[b]),
                    "stoich": int(stoich_np[b]),
                },
            )

            goal_align = 0.0
            if goal_props:
                diffs = [
                    abs(pred_props_dict[k] - float(v))
                    for k, v in goal_props.items()
                    if k in pred_props_dict
                ]
                if diffs:
                    goal_align += 1.0 - (sum(diffs) / len(diffs))
            if goal_caps:
                cap_hits = sum(
                    float(cap_probs[b, CAPABILITY_TO_IDX[c]])
                    for c in goal_caps
                    if c in CAPABILITY_TO_IDX
                )
                goal_align += 0.5 * (cap_hits / max(1, len(goal_caps)))

            novelty_bonus = sum(
                1.0 / math.sqrt(1.0 + it.times_used_in_valid_recipes) for it in subset
            ) / len(subset)
            op_ucb_bonus = 1.0 / math.sqrt(1.0 + op_counts.get(int(op1_np[b]), 0))

            pair_repeat_pen = 0.0
            sub_ids = tuple(sorted(it.item_id for it in subset))
            already_invented_count = existing_parent_sets.get(sub_ids, 0)
            if already_invented_count > 0:
                pair_repeat_pen += 6.0 * already_invented_count
            if len(subset) == 2:
                pk = tuple(sorted([subset[0].name, subset[1].name]))
                hist = self.pair_compatibility_memory.get(pk, [])
                if any(v >= 0.99 for v in hist):
                    pair_repeat_pen += 5.0
                elif len(hist) >= 2:
                    pair_repeat_pen += 2.5

            gate_bootstrap_bonus = 0.0
            if equipment_gates and not equipment_gates.get("wet_chemistry", {}).get("unlocked", True):
                if set(sub_ids) == {"elem_h", "elem_o"}:
                    gate_bootstrap_bonus = 8.0
            if any(it.tier >= 4 and it.category == "component" for it in subset) and already_invented_count == 0:
                gate_bootstrap_bonus += 2.5
            if excitation_field != "none" and already_invented_count == 0:
                gate_bootstrap_bonus += 5.5
                if any(it.item_id == "sub_photon" and it.times_used_in_valid_recipes == 0 for it in subset):
                    gate_bootstrap_bonus += 4.0

            score = (
                1.5 * float(val_probs[b])
                + 1.2 * float(compat_np[b])
                + 1.0 * float(phys_preds[b])
                + 1.8 * float(util_probs[b])
                + 0.4 * float(rew_preds[b])
                + 1.1 * goal_align
                + 0.8 * novelty_bonus
                + 0.9 * op_ucb_bonus
                + gate_bootstrap_bonus
                - pair_repeat_pen
            )

            eq_used = self._resolve_equipment_used(
                op1_name=op1_name,
                op2_name=op2_name,
                temp_c=temp_c,
                press_atm=press_atm,
                atmos_name=atmos_name,
                equipment_gates=equipment_gates,
                excitation_field=excitation_field,
            )

            bp = Blueprint(
                blueprint_id=bp_id,
                proposer_id=proposer_id,
                strategy=strategy,
                target_goal=target_goal_desc,
                ingredients=ings,
                steps=steps,
                neural_prediction=neural_pred,
                equipment_used=eq_used,
            )
            scored_blueprints.append((score, bp))

        scored_blueprints.sort(key=lambda x: x[0], reverse=True)
        return [bp for _, bp in scored_blueprints[:top_k]]

    def record_experience(
        self,
        blueprint: Blueprint,
        items_used: List[LibraryItem],
        verdict_valid: bool,
        physics_score: float,
        true_properties: Dict[str, float],
        true_capabilities: List[str],
        reward: float,
        utility_score: float = 0.80,
    ) -> None:
        feats = np.zeros((MAX_INGREDIENTS, ITEM_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(MAX_INGREDIENTS, dtype=np.bool_)
        for j, it in enumerate(items_used[:MAX_INGREDIENTS]):
            feats[j] = extract_item_feature_vector(it)
            mask[j] = True

        pol_idx = blueprint.neural_prediction.policy_indices
        op1_i = int(
            pol_idx.get(
                "op1",
                OP_TO_IDX.get(blueprint.steps[0].operation, 0) if blueprint.steps else 0,
            )
        )
        op2_i = int(
            pol_idx.get(
                "op2",
                STEP2_TO_IDX.get(blueprint.steps[1].operation, 0)
                if len(blueprint.steps) > 1
                else 0,
            )
        )
        temp_i = int(pol_idx.get("temp", 1))
        press_i = int(pol_idx.get("press", 1))
        atmos_i = int(
            pol_idx.get(
                "atmos",
                ATMOS_TO_IDX.get(blueprint.steps[0].atmosphere, 1)
                if blueprint.steps
                else 1,
            )
        )
        stoich_i = int(pol_idx.get("stoich", 0))

        prop_vec = np.array(
            [float(true_properties.get(k, 0.2)) for k in PROPERTY_KEYS],
            dtype=np.float32,
        )
        cap_vec = np.zeros(NUM_CAPABILITIES, dtype=np.float32)
        for c in true_capabilities:
            if c in CAPABILITY_TO_IDX:
                cap_vec[CAPABILITY_TO_IDX[c]] = 1.0

        goal_vec = np.concatenate([prop_vec, cap_vec], axis=0)

        with self.lock:
            self.replay_buffer.append(
                {
                    "feats": feats,
                    "mask": mask,
                    "op1": op1_i,
                    "op2": op2_i,
                    "temp": temp_i,
                    "press": press_i,
                    "atmos": atmos_i,
                    "stoich": stoich_i,
                    "valid": 1.0 if verdict_valid else 0.0,
                    "compat": 1.0 if (verdict_valid or physics_score >= 0.45) else 0.0,
                    "physics": float(physics_score),
                    "utility": float(utility_score) if verdict_valid else 0.05,
                    "props": prop_vec,
                    "caps": cap_vec,
                    "goal": goal_vec,
                    "reward": float(reward),
                }
            )
            if len(self.replay_buffer) > 4000:
                self.replay_buffer.pop(0)

            if len(items_used) >= 2:
                pair = tuple(sorted([items_used[0].name, items_used[1].name]))
                self.pair_compatibility_memory.setdefault(pair, []).append(
                    1.0 if verdict_valid else float(physics_score)
                )

    def train_step(
        self, batch_size: int = 24, updates: int = 3
    ) -> Optional[Dict[str, float]]:
        with self.lock:
            if len(self.replay_buffer) < 4:
                return None

            self.model.train()
            metrics_accum: Dict[str, float] = {
                "loss_total": 0.0,
                "loss_validity": 0.0,
                "loss_compat": 0.0,
                "loss_properties": 0.0,
                "loss_capabilities": 0.0,
                "loss_policy": 0.0,
                "validity_acc": 0.0,
                "property_mae": 0.0,
            }

            for _ in range(updates):
                k = min(batch_size, len(self.replay_buffer))
                batch = random.sample(self.replay_buffer, k)

                t_feats = torch.tensor(
                    np.stack([x["feats"] for x in batch]), device=self.device
                )
                t_mask = torch.tensor(
                    np.stack([x["mask"] for x in batch]), device=self.device
                )
                t_op1 = torch.tensor(
                    [x["op1"] for x in batch], dtype=torch.long, device=self.device
                )
                t_op2 = torch.tensor(
                    [x["op2"] for x in batch], dtype=torch.long, device=self.device
                )
                t_temp = torch.tensor(
                    [x["temp"] for x in batch], dtype=torch.long, device=self.device
                )
                t_press = torch.tensor(
                    [x["press"] for x in batch], dtype=torch.long, device=self.device
                )
                t_atmos = torch.tensor(
                    [x["atmos"] for x in batch], dtype=torch.long, device=self.device
                )
                t_stoich = torch.tensor(
                    [x["stoich"] for x in batch], dtype=torch.long, device=self.device
                )

                t_valid = torch.tensor(
                    [x["valid"] for x in batch], dtype=torch.float32, device=self.device
                )
                t_compat = torch.tensor(
                    [x["compat"] for x in batch], dtype=torch.float32, device=self.device
                )
                t_phys = torch.tensor(
                    [x["physics"] for x in batch], dtype=torch.float32, device=self.device
                )
                t_util = torch.tensor(
                    [x.get("utility", 0.85 if x["valid"] > 0.5 else 0.05) for x in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                t_props = torch.tensor(
                    np.stack([x["props"] for x in batch]), device=self.device
                )
                t_caps = torch.tensor(
                    np.stack([x["caps"] for x in batch]), device=self.device
                )
                t_goal = torch.tensor(
                    np.stack([x["goal"] for x in batch]), device=self.device
                )
                t_rew = torch.tensor(
                    [x["reward"] for x in batch], dtype=torch.float32, device=self.device
                )

                self.optimizer.zero_grad()

                z_ing, compat_logit = self.model.encode_ingredients(t_feats, t_mask)
                loss_compat = F.binary_cross_entropy_with_logits(
                    compat_logit.squeeze(-1), t_compat
                )

                world = self.model.forward_world_model(
                    z_ing=z_ing,
                    op1_idx=t_op1,
                    op2_idx=t_op2,
                    temp_idx=t_temp,
                    press_idx=t_press,
                    atmos_idx=t_atmos,
                    stoich_idx=t_stoich,
                )

                loss_validity = F.binary_cross_entropy_with_logits(
                    world["validity_logit"], t_valid
                )
                loss_phys = F.mse_loss(world["physics_pred"], t_phys)
                loss_util = F.mse_loss(world["utility_prob"], t_util)
                loss_rew = F.smooth_l1_loss(world["reward_pred"], t_rew)

                sample_weights = (0.2 + 0.8 * t_valid).unsqueeze(-1)
                loss_props = (
                    F.smooth_l1_loss(world["prop_pred"], t_props, reduction="none")
                    * sample_weights
                ).mean()
                loss_caps = (
                    F.binary_cross_entropy_with_logits(
                        world["cap_logits"], t_caps, reduction="none"
                    )
                    * sample_weights
                ).mean()

                pol = self.model.forward_policy(z_ing, t_goal, chosen_op1=t_op1)
                adv = torch.clamp(t_rew, min=0.0) + 0.5 * t_valid
                nll_op1 = F.cross_entropy(pol["op1_logits"], t_op1, reduction="none")
                nll_op2 = F.cross_entropy(pol["op2_logits"], t_op2, reduction="none")
                nll_temp = F.cross_entropy(pol["temp_logits"], t_temp, reduction="none")
                nll_atmos = F.cross_entropy(
                    pol["atmos_logits"], t_atmos, reduction="none"
                )
                loss_policy = (
                    (nll_op1 + 0.4 * nll_op2 + 0.8 * nll_temp + 0.6 * nll_atmos) * adv
                ).mean()

                loss_total = (
                    1.2 * loss_validity
                    + 1.0 * loss_compat
                    + 1.0 * loss_phys
                    + 1.2 * loss_util
                    + 2.5 * loss_props
                    + 1.5 * loss_caps
                    + 0.4 * loss_rew
                    + 0.35 * loss_policy
                )

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                self.train_steps += 1

                with torch.no_grad():
                    pred_v = (world["validity_prob"] >= 0.5).float()
                    acc = (pred_v == t_valid).float().mean().item()
                    mae = torch.abs(world["prop_pred"] - t_props).mean().item()

                metrics_accum["loss_total"] += loss_total.item() / updates
                metrics_accum["loss_validity"] += loss_validity.item() / updates
                metrics_accum["loss_compat"] += loss_compat.item() / updates
                metrics_accum["loss_properties"] += loss_props.item() / updates
                metrics_accum["loss_capabilities"] += loss_caps.item() / updates
                metrics_accum["loss_policy"] += loss_policy.item() / updates
                metrics_accum["validity_acc"] += acc / updates
                metrics_accum["property_mae"] += mae / updates

            metrics_rounded = {k: round(v, 4) for k, v in metrics_accum.items()}
            metrics_rounded["train_steps"] = self.train_steps
            metrics_rounded["replay_size"] = len(self.replay_buffer)
            self.loss_history.append(metrics_rounded)
            return metrics_rounded

    def get_telemetry_summary(self) -> Dict[str, Any]:
        with self.lock:
            latest = self.loss_history[-1] if self.loss_history else {}
            top_pairs = sorted(
                [
                    (
                        f"{p[0]} + {p[1]}",
                        round(sum(scores) / len(scores), 3),
                        len(scores),
                    )
                    for p, scores in self.pair_compatibility_memory.items()
                ],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )[:10]
            return {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU",
                "train_steps": self.train_steps,
                "replay_buffer_size": len(self.replay_buffer),
                "latest_metrics": latest,
                "loss_history": self.loss_history[-40:],
                "top_compatible_pairs": top_pairs,
            }

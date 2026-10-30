"""Main CLI & Orchestrator for Inventor-Argon: Autonomous Neural Invention System."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

from inventor.dashboard import start_dashboard_server
from inventor.judge import DeepSeekJudge
from inventor.library import SharedLibrary
from inventor.neural_net import SharedBrain
from inventor.schemas import ExperienceRecord, LibraryItem
from inventor.workers import ParallelInventorSwarm


def format_top_properties(props: Dict[str, float], top_n: int = 4) -> str:
    sorted_p = sorted(props.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted_p)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Inventor-Argon: Parallel Neural-Symbolic Invention Engine"
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=8,
        help="Number of parallel discovery waves to run (each wave runs all workers concurrently)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent Inventor workers sharing the GPU neural network",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v4-flash",
        help="Primary OpenRouter DeepSeek Judge model ID",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset library and neural checkpoint to start fresh from primitives",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for live web dashboard (http://127.0.0.1:8050)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Keep the interactive Web UI running after waves finish",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep the autonomous discovery & neural learning loop running continuously in the background",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=16.0,
        help="Seconds between autonomous discovery waves when --continuous is enabled",
    )
    parser.add_argument(
        "--max-idle-waves",
        type=int,
        default=80,
        help="Automatically pause autonomous loop after N consecutive waves with 0 new useful discoveries (0 = never stop)",
    )
    args = parser.parse_args()

    if args.reset:
        for p in ("data/library_state.json", "data/novel_inventions_registry.json", "data/inventor_net.pt"):
            if os.path.exists(p):
                os.remove(p)

    print("=" * 88)
    print("  INVENTOR-ARGON : NEURAL-SYMBOLIC AUTONOMOUS INVENTION & DISCOVERY ENGINE")
    print("=" * 88)

    library = SharedLibrary(
        state_path="data/library_state.json",
        novel_registry_path="data/novel_inventions_registry.json",
        reset_on_start=args.reset,
    )
    brain = SharedBrain(checkpoint_path="data/inventor_net.pt")
    judge = DeepSeekJudge(primary_model=args.model)
    swarm = ParallelInventorSwarm(
        library=library, brain=brain, judge=judge, num_workers=args.workers
    )

    tel = brain.get_telemetry_summary()
    init_stats = library.get_summary_stats()
    panel_desc = ", ".join(f"{j['label']} ({j['primary']})" for j in judge.panel_configs)
    print(f"  [GPU Brain]   Device: {tel['device']} ({tel['gpu_name']})")
    print(f"  [Judge Panel] 3-Model Voting Panel (>=2/3 majority required): {panel_desc}")
    print(
        f"  [Swarm]       {args.workers} Parallel Inventor Workers sharing Library & Joined Neural Net"
    )
    print(
        f"  [Library]     Starting with {init_stats['total_library_items']} items "
        f"({init_stats['seed_primitives']} primitives, {init_stats['known_real_world_items']} known, "
        f"{init_stats['novel_frontier_items']} novel frontier)"
    )

    loop_state = {
        "continuous_active": bool(args.continuous),
        "autonomous_waves_completed": 0,
    }

    def status_provider() -> Dict[str, Any]:
        return {
            "library_stats": library.get_summary_stats(),
            "brain_telemetry": brain.get_telemetry_summary(),
            "judge_stats": judge.get_stats(),
            "items": [it.to_dict() for it in library.get_all_items()],
            "novel_inventions": library.get_novel_inventions(),
            "recent_experiences": [
                e.to_dict() for e in library.experiences[-150:]
            ],
            "continuous_active": loop_state["continuous_active"],
            "autonomous_waves_completed": loop_state["autonomous_waves_completed"],
        }

    def review_handler(body: Dict[str, Any]) -> bool:
        return library.update_escalation_review(
            item_id=str(body.get("item_id", "")),
            escalation_status=str(body.get("escalation_status", "VERIFIED_NOVEL_BREAKTHROUGH")),
            chief_scientist_verdict=str(body.get("chief_scientist_verdict", "")),
            reviewed_by=str(body.get("reviewed_by", "Antigravity (Chief Scientist Escalation)")),
        )

    def escalate_handler(item_id: str) -> bool:
        it = library.get_item(item_id)
        if not it:
            return False
        return library.update_escalation_review(
            item_id=item_id,
            escalation_status="PENDING_ESCALATION",
            chief_scientist_verdict="Escalated to Novel Frontier Review Queue for deep first-principles audit.",
            reviewed_by="User / Antigravity Escalation",
        )

    def wave_runner() -> None:
        swarm.run_parallel_wave()
        loop_state["autonomous_waves_completed"] += 1

    def toggle_continuous_handler() -> bool:
        loop_state["continuous_active"] = not loop_state["continuous_active"]
        return True

    srv = start_dashboard_server(
        status_provider,
        port=args.port,
        review_handler=review_handler,
        escalate_handler=escalate_handler,
        wave_runner=wave_runner,
        toggle_continuous_handler=toggle_continuous_handler,
        simulate_handler=library.run_simulation_for_item,
    )
    if srv:
        print(f"  [Dashboard]   Interactive Command Center UI running at http://127.0.0.1:{args.port}")
    print("=" * 88)

    t_start = time.time()

    for wave_idx in range(1, args.waves + 1):
        w_t0 = time.time()
        print(f"\n>>> WAVE {wave_idx}/{args.waves} | Launching {args.workers} Parallel Inventors...")

        wave_results = swarm.run_parallel_wave()
        w_elapsed = time.time() - w_t0

        for exp, new_item in wave_results:
            bp = exp.blueprint
            vd = exp.verdict
            recipe_str = bp.summary_recipe_str()
            pred_v = bp.neural_prediction.predicted_validity
            pred_c = bp.neural_prediction.predicted_compatibility
            jv = vd.judge_votes or {}
            vote_summary = jv.get("decision", "N/A")
            ballots = jv.get("ballots", {})
            ballot_str = " | ".join(
                f"{k}: {v.get('vote', 'NO')}({v.get('output_name') or v.get('reason', '')[:32]})[{v.get('novelty_classification', 'KNOWN')[:5]}]"
                for k, v in ballots.items()
            )

            if exp.is_new_discovery and new_item is not None:
                is_nov = new_item.novelty_classification == "NOVEL_FRONTIER"
                banner = (
                    "🌟 NOVEL FRONTIER INVENTION (ESCALATED FOR DEEP REVIEW)!"
                    if is_nov
                    else "📚 NEW KNOWN REAL-WORLD INVENTION!"
                )
                extra_audit = ""
                if is_nov and new_item.deep_audit:
                    da = new_item.deep_audit
                    extra_audit = (
                        f"\n      Math/Thermo: {da.get('thermodynamics_and_math', '')[:160]}"
                        f"\n      Novelty Why: {da.get('novelty_vs_known_world', '')[:160]}"
                        f"\n      Escalation : Status={new_item.escalation_status} | SciMathScore={da.get('science_math_score', 0.85)}"
                    )
                print(
                    f"  [+] {banner} [{bp.proposer_id} | {bp.strategy}] [{vote_summary}]\n"
                    f"      Output     : {new_item.name} ({new_item.formula_or_spec}) "
                    f"[Tier {new_item.tier}: {new_item.category} | Phase: {new_item.phase} | Track: {new_item.novelty_classification}]\n"
                    f"      Recipe     : {recipe_str}\n"
                    f"      Judge Votes: {ballot_str}\n"
                    f"      Description: {new_item.description}\n"
                    f"      Properties : {format_top_properties(new_item.properties)}\n"
                    f"      Capabilities: {new_item.capabilities}"
                    f"{extra_audit}\n"
                    f"      Neural Net : PredValid={pred_v:.2f}, PredCompat={pred_c:.2f} -> Reward=+{exp.reward:.2f}"
                )
            elif vd.valid:
                print(
                    f"  [~] Valid Alternative Synthesis [{bp.proposer_id}] [{vote_summary}]: {vd.output_name} ({vd.formula_or_spec})\n"
                    f"      Recipe     : {recipe_str} | Reward=+{exp.reward:.2f}\n"
                    f"      Judge Votes: {ballot_str}"
                )
            else:
                hint_note = vd.repair_hint.get("note", "") if vd.repair_hint else ""
                print(
                    f"  [-] Rejected Recipe [{bp.proposer_id}] [{vote_summary}] (PhysicsScore={vd.physics_score:.2f}, Reward={exp.reward:.2f}):\n"
                    f"      Attempted  : {recipe_str}\n"
                    f"      Judge Votes: {ballot_str}\n"
                    f"      Critique   : {vd.failure_reason[:130]}"
                    + (f"\n      Repair Hint: {hint_note[:120]}" if hint_note else "")
                )

        # Print Wave Summary & Neural Learning Progress
        st = library.get_summary_stats()
        bt = brain.get_telemetry_summary()
        lm = bt.get("latest_metrics", {})
        print(
            f"--- Wave {wave_idx} Complete ({w_elapsed:.1f}s) | "
            f"Library: {st['total_library_items']} items ({st['invented_items']} invented, Tiers={st['tier_counts']}) | "
            f"TechGates: {st['equipment_gates_unlocked']} | "
            f"HitRate: {st['hit_rate']*100:.1f}% | "
            f"Neural Steps: {bt['train_steps']} | "
            f"Loss: {lm.get('loss_total', 0.0):.3f} | "
            f"ValidAcc: {lm.get('validity_acc', 0.0)*100:.1f}% | "
            f"PropMAE: {lm.get('property_mae', 0.0):.3f} ---",
            flush=True,
        )

    total_elapsed = time.time() - t_start
    final_stats = library.get_summary_stats()
    final_brain = brain.get_telemetry_summary()
    final_judge = judge.get_stats()

    print("\n" + "=" * 88)
    print("  FINAL RUN SUMMARY")
    print("=" * 88)
    print(f"  Total Runtime          : {total_elapsed:.1f}s")
    print(f"  Total Library Items    : {final_stats['total_library_items']} (Primitives: {final_stats['seed_primitives']}, Known Real-World: {final_stats['known_real_world_items']}, Novel Frontier: {final_stats['novel_frontier_items']})")
    print(f"  Novel Escalations      : {final_stats['novel_frontier_items']} total ({final_stats['pending_escalation_reviews']} pending Antigravity review, {final_stats['verified_novel_breakthroughs']} verified)")
    print(f"  Tier Breakdown         : {final_stats['tier_counts']}")
    print(f"  Blueprint Hit Rate     : {final_stats['hit_rate']*100:.1f}% ({final_stats['valid_blueprints']}/{final_stats['total_evaluations']})")
    print(f"  Equipment Gates        : {final_stats['equipment_gates_unlocked']}")
    for gk, gv in final_stats.get("equipment_gates", {}).items():
        status_str = f"UNLOCKED by {gv['equipment']}" if gv["unlocked"] else "LOCKED"
        print(f"    - {gk:22s}: {status_str}")
    print(f"  Concept Families       : {final_stats.get('concept_families', {})}")
    print(f"  Capabilities Unlocked  : {len(final_stats['capabilities_unlocked'])}/24 -> {', '.join(final_stats['capabilities_unlocked'])}")
    print(f"  Neural Training Steps  : {final_brain['train_steps']} on {final_brain['gpu_name']}")
    print(f"  Top Compatible Pairs   : {final_brain['top_compatible_pairs'][:5]}")
    print(f"  3-Judge Panel Stats    : {final_judge}")
    print("=" * 88)

    if args.serve and srv:
        print(
            f"\n  [UI Server Active] Open http://127.0.0.1:{args.port} in your browser to inspect inventions & escalation reviews."
        )
        if args.continuous:
            print(
                f"  [Autonomous Loop] Continuous Discovery & GPU Learning Loop ACTIVE (running a 5-worker wave every {args.interval}s)...",
                flush=True,
            )
        idle_waves = 0
        try:
            while True:
                if loop_state["continuous_active"]:
                    w_num = loop_state["autonomous_waves_completed"] + 1
                    print(
                        f"\n>>> [AUTONOMOUS LOOP WAVE #{w_num} | Idle={idle_waves}/{args.max_idle_waves or '∞'}] Launching {args.workers} Parallel Inventors...",
                        flush=True,
                    )
                    try:
                        w_res = swarm.run_parallel_wave()
                        loop_state["autonomous_waves_completed"] = w_num
                        found_useful_this_wave = 0
                        for exp, new_item in w_res:
                            if exp.is_new_discovery and new_item is not None:
                                if new_item.utility_score >= 0.80 and new_item.escalation_status != "REJECTED_DEEP_AUDIT":
                                    found_useful_this_wave += 1
                                sim_tag = ""
                                if new_item.simulation_verification:
                                    sp = new_item.simulation_verification.get("simulation_passed")
                                    sim_tag = f" | 🔬 GPU Sim: {'PASSED' if sp else 'FAILED'}"
                                print(
                                    f"  [+] NEW AUTONOMOUS INVENTION: {new_item.name} ({new_item.formula_or_spec}) "
                                    f"[Tier {new_item.tier}: {new_item.category} | Track: {new_item.novelty_classification} | "
                                    f"Real-World Equivalent: {new_item.real_world_equivalent} (Utility={int(new_item.utility_score*100)}%){sim_tag}]",
                                    flush=True,
                                )
                        if found_useful_this_wave > 0:
                            idle_waves = 0
                        else:
                            idle_waves += 1
                            if args.max_idle_waves > 0 and idle_waves >= args.max_idle_waves:
                                loop_state["continuous_active"] = False
                                print(
                                    f"\n  [✓] DISCOVERY SATURATION REACHED: {idle_waves} consecutive waves completed with 0 new useful inventions. "
                                    f"Autonomous loop paused automatically (Web UI remains active at http://127.0.0.1:{args.port}).",
                                    flush=True,
                                )
                    except Exception as wave_exc:
                        print(f"  [!] Autonomous wave #{w_num} transient error (continuing): {wave_exc}", flush=True)
                    time.sleep(max(3.0, args.interval))
                else:
                    time.sleep(1.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

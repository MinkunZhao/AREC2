"""TextGrad engine: offline prompt optimization via textual gradients."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from arec2.agents.llm_client import LLMClient
from arec2.textgrad.meta_prompts import CRITIC_META, OPTIMIZER_META
from arec2.textgrad.text_module import TextVariable

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A single forward pass trajectory with reward."""
    uid: int
    task_type: str
    history_summary: str
    plan_json: dict  # Tool calls as JSON
    card: str  # Compiled context card
    pred: str  # Model prediction (SID string)
    gt: str  # Ground truth (SID string)
    reward: float  # R@10 or other metric


class TextGradEngine:
    """Offline prompt optimizer using textual gradients.

    Iteration loop:
      1. Run forward pass on a held-out dev minibatch to collect Trajectories.
      2. Sort by reward; keep top-4 + bottom-4.
      3. Critic generates NL critiques for planner and compiler separately.
      4. Optimizer rewrites planner_instruction and compiler_instruction.
      5. Validate the rewritten instructions on a held-out validation set.
      6. Only commit if validation reward improves (or stays within threshold).
      7. Save versioned snapshots.
    """

    def __init__(
        self,
        llm: LLMClient,
        forward_fn,  # callable: (planner_instr, compiler_instr, batch) -> [Trajectory]
        save_dir: str = "prompts/optimized",
        n_iterations: int = 5,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        commit_threshold_pp: float = -0.5,  # Accept if val drop <= 0.5pp
    ):
        """Initialize TextGrad engine.

        Args:
            llm: LLM client for critic and optimizer.
            forward_fn: Callable that runs forward pass and returns trajectories.
            save_dir: Directory to save optimized prompts.
            n_iterations: Number of optimization iterations.
            train_batch_size: Size of training batch per iteration.
            val_batch_size: Size of validation batch.
            commit_threshold_pp: Minimum improvement (in percentage points) to accept new prompts.
        """
        self.llm = llm
        self.forward_fn = forward_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_iterations = n_iterations
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.commit_threshold_pp = commit_threshold_pp
        logger.info(
            f"TextGradEngine initialized | iterations={n_iterations} | "
            f"train_batch={train_batch_size} | val_batch={val_batch_size}"
        )

    def optimize(
        self,
        planner_instruction: TextVariable,
        compiler_instruction: TextVariable,
        train_samples: list,
        val_samples: list,
    ) -> tuple[TextVariable, TextVariable, list[dict]]:
        """Run optimization loop.

        Args:
            planner_instruction: Initial planner instruction.
            compiler_instruction: Initial compiler instruction.
            train_samples: Training samples for optimization.
            val_samples: Validation samples for evaluation.

        Returns:
            Tuple of (optimized_planner, optimized_compiler, history).
        """
        history = []
        best_plan = planner_instruction.value
        best_comp = compiler_instruction.value

        # Initial validation
        logger.info("Running initial validation...")
        best_val = self._evaluate(best_plan, best_comp, val_samples)
        logger.info(f"[TextGrad] Initial val R@10 = {best_val:.4f}")

        for it in range(self.n_iterations):
            t0 = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"TextGrad Iteration {it+1}/{self.n_iterations}")
            logger.info(f"{'='*60}")

            # Sample training batch
            batch = random.sample(train_samples, min(self.train_batch_size, len(train_samples)))

            # Forward pass
            logger.info(f"Running forward pass on {len(batch)} samples...")
            trajs = self.forward_fn(best_plan, best_comp, batch)
            trajs.sort(key=lambda x: x.reward, reverse=True)
            top, bot = trajs[:4], trajs[-4:]

            logger.info(f"Top-4 rewards: {[t.reward for t in top]}")
            logger.info(f"Bottom-4 rewards: {[t.reward for t in bot]}")

            # Compute textual gradient (critique)
            logger.info("Generating critiques...")
            critique_obj = self._critic(best_plan, best_comp, top, bot)
            logger.info(f"Planner critiques: {len(critique_obj['planner_critiques'])}")
            logger.info(f"Compiler critiques: {len(critique_obj['compiler_critiques'])}")

            # Optimize instructions
            logger.info("Optimizing instructions...")
            new_plan = self._optimize_one(best_plan, critique_obj["planner_critiques"])
            new_comp = self._optimize_one(best_comp, critique_obj["compiler_critiques"])

            # Validate
            logger.info("Validating new instructions...")
            new_val = self._evaluate(new_plan, new_comp, val_samples)
            improvement_pp = (new_val - best_val) * 100
            accepted = improvement_pp >= self.commit_threshold_pp

            elapsed = time.time() - t0
            logger.info(
                f"[TextGrad iter {it+1}] new_val={new_val:.4f} | best_val={best_val:.4f} | "
                f"improvement={improvement_pp:+.2f}pp | accepted={accepted} | "
                f"took={elapsed:.1f}s | spend=${self.llm.total_cost:.3f}"
            )

            if accepted:
                logger.info("Accepting new instructions.")
                best_plan, best_comp, best_val = new_plan, new_comp, new_val
            else:
                logger.info("Rejecting new instructions (insufficient improvement).")

            # Save snapshot
            self._snapshot(it, best_plan, best_comp, new_val, accepted, critique_obj)
            history.append({
                "iter": it,
                "val": new_val,
                "improvement_pp": improvement_pp,
                "accepted": accepted,
            })

        # Save final
        logger.info(f"\n{'='*60}")
        logger.info("Optimization complete. Saving final prompts...")
        (self.save_dir / "planner_final.txt").write_text(best_plan, encoding="utf-8")
        (self.save_dir / "compiler_final.txt").write_text(best_comp, encoding="utf-8")
        logger.info(f"Final prompts saved to {self.save_dir}")
        logger.info(f"Final val R@10: {best_val:.4f}")
        logger.info(f"Total cost: ${self.llm.total_cost:.3f}")
        logger.info(f"{'='*60}\n")

        return (
            TextVariable(name="planner_instruction", value=best_plan, requires_grad=False),
            TextVariable(name="compiler_instruction", value=best_comp, requires_grad=False),
            history,
        )

    def _critic(
        self,
        plan_instr: str,
        comp_instr: str,
        top: list[Trajectory],
        bot: list[Trajectory],
    ) -> dict:
        """Generate critiques by comparing top and bottom trajectories."""
        # Build trajectory table
        traj_table = "HIGH-REWARD TRAJECTORIES:\n"
        for i, t in enumerate(top, 1):
            traj_table += f"\n[Top-{i}] Reward={t.reward:.3f}\n"
            traj_table += f"  Task: {t.task_type}\n"
            traj_table += f"  History: {t.history_summary}\n"
            traj_table += f"  Plan: {json.dumps(t.plan_json, ensure_ascii=False)}\n"
            traj_table += f"  Card (first 200 chars): {t.card[:200]}...\n"
            traj_table += f"  Pred: {t.pred}\n"
            traj_table += f"  GT: {t.gt}\n"

        traj_table += "\n\nLOW-REWARD TRAJECTORIES:\n"
        for i, t in enumerate(bot, 1):
            traj_table += f"\n[Bottom-{i}] Reward={t.reward:.3f}\n"
            traj_table += f"  Task: {t.task_type}\n"
            traj_table += f"  History: {t.history_summary}\n"
            traj_table += f"  Plan: {json.dumps(t.plan_json, ensure_ascii=False)}\n"
            traj_table += f"  Card (first 200 chars): {t.card[:200]}...\n"
            traj_table += f"  Pred: {t.pred}\n"
            traj_table += f"  GT: {t.gt}\n"

        user_msg = (
            f"PLANNER INSTRUCTION:\n```\n{plan_instr}\n```\n\n"
            f"COMPILER INSTRUCTION:\n```\n{comp_instr}\n```\n\n"
            f"{traj_table}\n\n"
            "Analyze the differences and return JSON with planner_critiques and compiler_critiques."
        )

        messages = [
            {"role": "system", "content": CRITIC_META},
            {"role": "user", "content": user_msg},
        ]

        resp = self.llm.chat(messages, temperature=0.0, json_mode=True)
        if resp.json_obj is None:
            logger.warning("Critic did not return valid JSON. Using empty critiques.")
            return {"planner_critiques": [], "compiler_critiques": []}

        return resp.json_obj

    def _optimize_one(self, current_instruction: str, critiques: list[str]) -> str:
        """Apply critiques to optimize an instruction."""
        if not critiques:
            logger.debug("No critiques, keeping instruction unchanged.")
            return current_instruction

        critique_text = "\n".join(f"- {c}" for c in critiques)
        user_msg = (
            f"CURRENT INSTRUCTION:\n```\n{current_instruction}\n```\n\n"
            f"CRITIQUES:\n{critique_text}\n\n"
            "Produce a revised instruction that addresses these critiques."
        )

        messages = [
            {"role": "system", "content": OPTIMIZER_META},
            {"role": "user", "content": user_msg},
        ]

        resp = self.llm.chat(messages, temperature=0.0)
        revised = resp.text.strip()

        # Remove markdown fences if present
        if revised.startswith("```"):
            lines = revised.split("\n")
            revised = "\n".join(lines[1:-1]) if len(lines) > 2 else revised

        return revised

    def _evaluate(self, plan_instr: str, comp_instr: str, samples: list) -> float:
        """Evaluate instructions on validation samples."""
        trajs = self.forward_fn(plan_instr, comp_instr, samples)
        avg_reward = sum(t.reward for t in trajs) / len(trajs) if trajs else 0.0
        return avg_reward

    def _snapshot(
        self,
        it: int,
        plan: str,
        comp: str,
        val: float,
        accepted: bool,
        critique: dict,
    ):
        """Save iteration snapshot."""
        d = self.save_dir / f"iter_{it:02d}"
        d.mkdir(exist_ok=True, parents=True)
        (d / "planner.txt").write_text(plan, encoding="utf-8")
        (d / "compiler.txt").write_text(comp, encoding="utf-8")
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "val_R10": val,
                    "accepted": accepted,
                    "critiques": critique,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

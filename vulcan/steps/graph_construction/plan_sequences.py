"""plan_sequences step: Subgraph extraction + sequence construction (deterministic).

Two independent graph processing approaches:
  1. Good subgraphs → sequences (chain patterns for multi-turn)
  2. Multipath subgraphs (diamond/fork-merge patterns with multiple paths)

Both produce typed, uniquely-IDed outputs that are merged at the end.

Config flags:
  enable_multipath: true/false — whether to extract multipath subgraphs (default: False)
"""

from __future__ import annotations

import os
from collections import defaultdict

from .. import register_step
from ..base import BaseStep
from ...core.io import load_jsonl, save_jsonl, ensure_dir
from ._subgraph_extractor import extract_good_subgraphs, sample_by_popularity
from ._multipath_extractor import extract_multipath_subgraphs
from ._sequence_builder import build_sequences


@register_step("plan_sequences")
class PlanSequencesStep(BaseStep):
    """Extract good subgraphs, build multi-turn sequences, and optionally extract
    multipath subgraphs. Writes results to multi_turn/ inside output_dir.

    Output files:
        {output_dir}/good_graph_{cat}.json      — per-category good subgraphs
        {output_dir}/sequences_{cat}.jsonl      — per-category sequences
        {output_dir}/multi_turn/sequences_only.jsonl — sequences, for inspection
        {output_dir}/multi_turn/all_multipath.jsonl  — only when enable_multipath=True
        {output_dir}/multi_turn/all_sequences.jsonl  — merged; THIS is what the
                                                      next stage reads
    """

    requires_llm = False

    async def run(self, inp: dict, progress=None) -> dict:
        """No-op: this step uses run_batch() instead."""
        return inp

    def run_batch(self, input_path: str, output_dir: str) -> None:
        """Main entry point: extract subgraphs and build sequences.

        Args:
            input_path: path to build_graph success.jsonl
            output_dir: directory to write all outputs
        """
        ensure_dir(output_dir)
        data = load_jsonl(input_path)

        # -- Config --
        max_size = self.step_config.get("max_tools_per_sequence", 5)
        implicit_edge_weight = self.step_config.get("implicit_edge_weight", 0.01)
        turns_per_sequence_str = self.step_config.get("turns_per_sequence", "2,3,4")
        if isinstance(turns_per_sequence_str, str):
            turns_per_sequence = [int(k) for k in turns_per_sequence_str.split(",")]
        else:
            turns_per_sequence = list(turns_per_sequence_str)
        sequences_per_category = self.step_config.get("sequences_per_category", 15000)
        target_per_k = sequences_per_category // max(len(turns_per_sequence), 1)
        enable_multipath = self.step_config.get("enable_multipath", False)

        # Collect all unique categories
        categories: set[str] = set()
        for item in data:
            cat = item.get("category_name")
            if cat:
                categories.add(cat)

        id_counter = 0  # global unique ID across all types

        # ══════════════════════════════════════════════════════════════
        # APPROACH 1: Good subgraphs → Sequences
        # ══════════════════════════════════════════════════════════════

        # ── Phase 1a: Extract good subgraphs per category ─────────────
        for cat in sorted(categories):
            print(f"\n[plan_sequences] Phase 1a: good subgraphs for {cat}")
            cat_data = [d for d in data if d.get("category_name") == cat]
            if not cat_data:
                continue

            graph_entries = []
            catalog = []
            for d in cat_data:
                graph_entries.extend(d.get("graph", []))
                catalog.extend(d.get("api_list", []))

            good_subgraphs = extract_good_subgraphs(
                graph_entries, catalog,
                min_size=2, max_size=max_size, implicit_edge_weight=implicit_edge_weight,
            )

            max_per_cat = self.step_config.get("max_subgraphs_per_category", 5000)
            if len(good_subgraphs) > max_per_cat:
                good_subgraphs = sample_by_popularity(good_subgraphs, max_per_cat, implicit_edge_weight)
                print(f"    Sampled to {len(good_subgraphs)}")

            good_path = os.path.join(output_dir, f"good_graph_{cat}.json")
            save_jsonl(good_subgraphs, good_path)

        # ── Phase 1b: Build sequences from good subgraphs ─────────────
        all_sequences: list[dict] = []

        for cat in sorted(categories):
            good_path = os.path.join(output_dir, f"good_graph_{cat}.json")
            if not os.path.exists(good_path):
                continue

            good_subgraphs = load_jsonl(good_path)
            if not good_subgraphs:
                continue

            cat_data = [d for d in data if d.get("category_name") == cat]
            all_graph_edges = self._extract_all_edges(cat_data)
            catalog = []
            for d in cat_data:
                catalog.extend(d.get("api_list", []))

            print(f"[plan_sequences] Phase 1b: building sequences for {cat} ({len(good_subgraphs)} subgraphs)")

            sequences = build_sequences(
                good_subgraphs=good_subgraphs,
                all_graph_edges=all_graph_edges,
                catalog=catalog,
                turns_per_sequence=turns_per_sequence,
                target_per_k=target_per_k,
                implicit_edge_weight=0.5,
            )

            # Assign type, unique ID, and category
            for seq in sequences:
                seq["id"] = f"seq_{id_counter}"
                seq["type"] = "sequence"
                seq["category_name"] = cat
                id_counter += 1

            cat_out_path = os.path.join(output_dir, f"sequences_{cat}.jsonl")
            save_jsonl(sequences, cat_out_path)
            print(f"    {len(sequences)} sequences")

            all_sequences.extend(sequences)

        # Dedup prefixes, then balance and save sequences
        mt_dir = os.path.join(output_dir, "multi_turn")
        ensure_dir(mt_dir)
        if all_sequences:
            before_dedup = len(all_sequences)
            all_sequences = self._dedup_prefixes(all_sequences)
            print(
                f"\n[plan_sequences] Prefix dedup: {before_dedup} → {len(all_sequences)} "
                f"({before_dedup - len(all_sequences)} removed)"
            )
            sampled_seqs = self._balanced_sample(all_sequences, sequences_per_category, turns_per_sequence)
            # Sequences only, for inspection. The file the next stage reads is the
            # merged one written at the end of this method — keep the two names
            # distinct, or the merge silently truncates this artifact.
            save_jsonl(sampled_seqs, os.path.join(mt_dir, "sequences_only.jsonl"))
            print(f"[plan_sequences] Sequences total: {len(sampled_seqs)} (sampled from {len(all_sequences)})")
        else:
            sampled_seqs = []

        # ══════════════════════════════════════════════════════════════
        # APPROACH 2: Multipath subgraphs (independent)
        # ══════════════════════════════════════════════════════════════

        all_multipath: list[dict] = []

        if enable_multipath:
            for cat in sorted(categories):
                cat_data = [d for d in data if d.get("category_name") == cat]
                if not cat_data:
                    continue

                graph_entries = []
                for d in cat_data:
                    graph_entries.extend(d.get("graph", []))

                print(f"\n[plan_sequences] Phase 2: multipath subgraphs for {cat}")

                max_mp_size = self.step_config.get("max_multipath_tools", 15)
                multipath = extract_multipath_subgraphs(
                    graph_entries,
                    max_tools_per_sequence=max_mp_size,
                )

                # Assign type, unique ID, and category
                for sg in multipath:
                    sg["id"] = f"mp_{id_counter}"
                    sg["type"] = "multipath"
                    sg["category_name"] = cat
                    id_counter += 1

                if multipath:
                    mp_path = os.path.join(output_dir, f"multipath_{cat}.jsonl")
                    save_jsonl(multipath, mp_path)

                all_multipath.extend(multipath)

            if all_multipath:
                save_jsonl(all_multipath, os.path.join(mt_dir, "all_multipath.jsonl"))
                print(f"\n[plan_sequences] Multipath total: {len(all_multipath)}")
        else:
            print("\n[plan_sequences] Multipath extraction: DISABLED")

        # ══════════════════════════════════════════════════════════════
        # MERGE: Combined output with both types
        # ══════════════════════════════════════════════════════════════

        # This is the file execute_sequences and score_sequences read
        # (stage_runner._PLAN_SEQUENCES_OUT). It holds sequences plus, when
        # enable_multipath is on, the multipath subgraphs.
        merged = sampled_seqs + all_multipath
        merged_path = os.path.join(mt_dir, "all_sequences.jsonl")
        save_jsonl(merged, merged_path)

        seq_count = len(sampled_seqs)
        mp_count = len(all_multipath)
        print(
            f"\n[plan_sequences] MERGED: {len(merged)} total "
            f"({seq_count} sequences + {mp_count} multipath) → {merged_path}"
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _extract_all_edges(self, cat_data: list[dict]) -> list[dict]:
        """Flatten all graph edges from a list of category items."""
        all_edges: list[dict] = []
        for item in cat_data:
            for g in item.get("graph", []):
                source = g.get("source")
                for edge_info in g.get("all_edges", []):
                    target = edge_info.get("target")
                    edges = edge_info.get("edges")
                    if edges and isinstance(edges, dict) and edges.get("type") in ("explicit", "implicit"):
                        all_edges.append({
                            "source": source,
                            "target": target,
                            "type": edges["type"],
                            "arguments": edges.get("arguments", {}),
                            "source_arguments": edges.get("source_arguments", []),
                            "target_arguments": edges.get("target_arguments", []),
                        })
        return all_edges

    def _dedup_prefixes(self, sequences: list[dict]) -> list[dict]:
        """Remove sequences that are prefixes of longer sequences.

        A 2-turn [A, B] is redundant if a 3-turn [A, B, C] exists with
        the same first two turns. We keep the longer one.

        Algorithm: O(N × k) hash-based prefix detection.
        1. Sort by turns descending (longest first)
        2. For each kept sequence, register all strict prefixes as covered
        3. Skip any sequence whose signature matches a covered prefix
        """
        def _signature(seq: dict):
            turns = seq.get("sequence", [])
            return tuple(
                frozenset(t.get("graph_info", t).get("nodes", []))
                for t in turns
            )

        sequences.sort(key=lambda s: -s.get("number_turns", 0))

        covered: set = set()
        kept: list[dict] = []

        for seq in sequences:
            sig = _signature(seq)
            if not sig:
                continue
            if sig in covered:
                continue  # this sequence is a prefix of a longer one

            kept.append(seq)

            # Register all strict prefixes as covered
            for prefix_len in range(1, len(sig)):
                covered.add(sig[:prefix_len])

        return kept

    def _balanced_sample(
        self,
        sequences: list[dict],
        target: int,
        turns_per_sequence: list[int],
    ) -> list[dict]:
        """Sample sequences balancing across (category, number_turns) groups."""
        groups: dict = defaultdict(list)
        for seq in sequences:
            key = (seq.get("category_name", ""), seq.get("number_turns", 0))
            groups[key].append(seq)

        per_group = max(1, target // max(len(groups), 1))
        sampled: list[dict] = []
        for key in sorted(groups.keys()):
            sampled.extend(groups[key][:per_group])

        return sampled[:target]

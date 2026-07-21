from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib

# Use the Agg backend so chart rendering never requires a display server
# (e.g., in CI or headless environments). Must be set before importing
# pyplot to avoid RuntimeError from Matplotlib's backend lock.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enrichment_router.budget import Budget
from enrichment_router.graph import run_enrichment
from enrichment_router.models import EnrichmentResult
from enrichment_router.tools.llm import LLMClient
from enrichment_router.validation import MissingNameError, validate_request
from eval.baseline import load_dataset, run_baseline

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


@dataclass
class StrategyMetrics:
    name: str
    total_cost_usd: float
    total_latency_ms: float
    total_fields_needed: int
    total_fields_resolved: int
    completion_rate: float
    records_processed: int
    records_fully_resolved: int


@dataclass
class EvalReport:
    router: StrategyMetrics
    baseline: StrategyMetrics
    cost_savings_pct: float
    latency_savings_pct: float


def _aggregate(name: str, results: list[EnrichmentResult]) -> StrategyMetrics:
    total_needed = sum(len(r.request.fields_needed) for r in results)
    total_resolved = sum(len(r.resolved) for r in results)
    fully = sum(1 for r in results if r.status == "done_all_resolved")
    return StrategyMetrics(
        name=name,
        total_cost_usd=sum(r.total_cost_usd for r in results),
        total_latency_ms=sum(r.total_latency_ms for r in results),
        total_fields_needed=total_needed,
        total_fields_resolved=total_resolved,
        completion_rate=(total_resolved / total_needed) if total_needed else 1.0,
        records_processed=len(results),
        records_fully_resolved=fully,
    )


def run_eval(
    llm_client: LLMClient,
    wiki_fetcher: Optional[Callable] = None,
) -> EvalReport:
    """Run both router and baseline strategies over the full dataset.

    Both strategies share the same LLM client so that token pricing
    and latency measurements are directly comparable — a different
    model or provider between strategies would make the cost/latency
    comparison meaningless. The router also receives the optional
    wiki_fetcher (the baseline ignores it by design).
    """
    dataset = load_dataset()
    router_results: list[EnrichmentResult] = []
    baseline_results: list[EnrichmentResult] = []

    for raw in dataset:
        try:
            req = validate_request(raw)
        except MissingNameError:
            continue

        budget = Budget(
            max_cost_usd=raw.get("max_cost_usd", 0.05),
            max_latency_ms=raw.get("max_latency_ms", 10000.0),
        )
        result, _trace = run_enrichment(
            request=req,
            budget=budget,
            llm_client=llm_client,
            wiki_fetcher=wiki_fetcher,
        )
        router_results.append(result)
        baseline_results.append(run_baseline(raw, llm_client))

    router_m = _aggregate("router", router_results)
    baseline_m = _aggregate("baseline", baseline_results)
    cost_savings = (
        (baseline_m.total_cost_usd - router_m.total_cost_usd) / baseline_m.total_cost_usd * 100
        if baseline_m.total_cost_usd > 0
        else 0.0
    )
    latency_savings = (
        (baseline_m.total_latency_ms - router_m.total_latency_ms)
        / baseline_m.total_latency_ms
        * 100
        if baseline_m.total_latency_ms > 0
        else 0.0
    )
    return EvalReport(
        router=router_m,
        baseline=baseline_m,
        cost_savings_pct=cost_savings,
        latency_savings_pct=latency_savings,
    )


def render_chart(report: EvalReport, output_path: Path | str) -> None:
    """Render a side-by-side comparison chart as a PNG.

    Uses twin axes (ax1 for cost in USD, ax2 for latency in ms) so
    both metrics can be compared across strategies in a single chart
    without one dwarfing the other due to scale differences.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    labels = ["Router", "Baseline"]
    x = range(len(labels))
    width = 0.35
    costs = [report.router.total_cost_usd, report.baseline.total_cost_usd]
    latencies = [report.router.total_latency_ms, report.baseline.total_latency_ms]
    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        costs,
        width,
        label="Total cost (USD)",
        color="#4C72B0",
    )
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        latencies,
        width,
        label="Total latency (ms)",
        color="#DD8452",
    )
    ax1.set_ylabel("Total modeled cost (USD)")
    ax2.set_ylabel("Total latency (ms)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_title("Router vs Baseline: cost and latency over 18-record eval set")
    for bar, val in zip(bars1, costs):
        ax1.annotate(
            f"${val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val in zip(bars2, latencies):
        ax2.annotate(
            f"{val:.0f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _print_report(report: EvalReport) -> None:
    print("=" * 60)
    print("Eval report: router vs baseline")
    print("=" * 60)
    for m in (report.router, report.baseline):
        print(f"\n{m.name}:")
        print(f"  records processed:       {m.records_processed}")
        print(f"  records fully resolved:  {m.records_fully_resolved}")
        print(f"  total fields needed:     {m.total_fields_needed}")
        print(f"  total fields resolved:   {m.total_fields_resolved}")
        print(f"  completion rate:         {m.completion_rate:.1%}")
        print(f"  total modeled cost:      ${m.total_cost_usd:.6f}")
        print(f"  total latency:           {m.total_latency_ms:.1f} ms")
    print(f"\nCost savings:   {report.cost_savings_pct:.1f}%")
    print(f"Latency savings: {report.latency_savings_pct:.1f}%")


def main() -> None:
    """Run the eval against the live Groq LLM provider.

    Requires the GROQ_API_KEY environment variable. Groq's free tier
    has rate limits — the 18-record dataset should fit within them,
    but if you exceed limits, wait a few minutes and re-run.
    """
    from enrichment_router.tools.llm import DEFAULT_GROQ_CONFIG, GroqLLMClient

    client = GroqLLMClient(DEFAULT_GROQ_CONFIG)
    report = run_eval(client)
    _print_report(report)
    output = _REPO_ROOT / "assets" / "cost_savings.png"
    render_chart(report, output)
    print(f"\nChart written to {output}")


if __name__ == "__main__":
    main()

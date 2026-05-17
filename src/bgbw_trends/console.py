"""Rich console and formatting utilities for the BGBW trends pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Shared console ────────────────────────────────────────────────────────────

_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "step": "bold white",
        "metric": "cyan",
        "value": "white",
        "good": "green",
        "neutral": "dim white",
    }
)

console = Console(theme=_THEME, highlight=False)


# ── Logging integration ───────────────────────────────────────────────────────

def setup_rich_logging(level: int = logging.INFO) -> None:
    """Replace the root logging handler with a Rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                console=console,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            )
        ],
        force=True,
    )


# ── Section headers ───────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a section rule."""
    console.print(Rule(f"[step]{title}[/step]", style="dim"))


# ── Progress bar factory ──────────────────────────────────────────────────────

def make_progress() -> Progress:
    """Standard pipeline progress bar with spinner, bar, counts, and time."""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[step]{task.description}[/step]"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def make_spinner() -> Progress:
    """Minimal spinner for tasks of unknown duration."""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[step]{task.description}[/step]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# ── Timed spinner context ─────────────────────────────────────────────────────

@contextmanager
def timed_step(description: str) -> Generator[None, None, None]:
    """Context manager: show a spinner while work runs, then print elapsed."""
    spinner = make_spinner()
    task = spinner.add_task(description, total=None)
    t0 = time.perf_counter()
    with spinner:
        yield
    elapsed = time.perf_counter() - t0
    console.print(f"  [dim]done in {elapsed:.1f}s[/dim]")


# ── Result tables ─────────────────────────────────────────────────────────────

def print_trend_table(
    rows: Sequence[tuple[str, str, float, float, float, float, float]],
    title: str = "Global trend estimates",
) -> None:
    """Print a formatted table of DML trend results.

    Each row: (species, outcome, tau, se, pvalue, ci_lower, ci_upper)
    """
    table = Table(title=title, show_header=True, header_style="bold dim")
    table.add_column("Species", style="", no_wrap=True)
    table.add_column("Outcome", style="dim")
    table.add_column("tau", justify="right", style="cyan")
    table.add_column("SE", justify="right", style="dim")
    table.add_column("p-value", justify="right")
    table.add_column("95% CI", justify="right", style="dim")
    table.add_column("R2 ml_l", justify="right", style="dim")
    table.add_column("R2 ml_m", justify="right", style="dim")

    for row in rows:
        sp, outcome, tau, se, pval, ci_lo, ci_hi, r2_l, r2_m = row
        pval_str = Text(f"{pval:.4f}")
        if pval < 0.001:
            pval_str.stylize("bold green")
        elif pval < 0.05:
            pval_str.stylize("green")
        table.add_row(
            sp,
            outcome,
            f"{tau:+.4f}",
            f"{se:.4f}",
            pval_str,
            f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
            f"{r2_l:.3f}",
            f"{r2_m:.3f}",
        )

    console.print(table)


def print_sdm_table(rows: Sequence[tuple], title: str = "Pipeline RF diagnostics") -> None:
    """Print pipeline RF diagnostic metrics.

    Each row: (species, role, metric, value).  ``role`` should be one of the
    DML stage labels ("marginal", "propensity").  Propensity R² values are
    colour-coded: dim = low confounding (DML buying little), yellow = moderate,
    red = very high (check for year-proxy features in X).
    """
    table = Table(title=title, show_header=True, header_style="bold dim")
    table.add_column("Species", no_wrap=True)
    table.add_column("Forest", style="dim")
    table.add_column("Metric", justify="right", style="cyan")
    table.add_column("Value", justify="right")

    for sp, role, metric, value in rows:
        is_propensity_r2 = "propensity" in role and metric == "R²"
        if is_propensity_r2:
            if value > 0.85:
                style = "bold red"
            elif value > 0.4:
                style = "yellow"
            else:
                style = "dim"
            val_text = Text(f"{value:.4f}", style=style)
        else:
            val_text = Text(f"{value:.4f}")
        table.add_row(sp, role, metric, val_text)

    console.print(table)


def print_spatial_summary(
    species: str,
    tau_mean: float,
    tau_min: float,
    tau_max: float,
) -> None:
    """Print a one-line spatial trend summary."""
    console.print(
        f"  [dim]tau:[/dim] mean [cyan]{tau_mean:+.4f}[/cyan]"
        f"  range [[cyan]{tau_min:+.4f}[/cyan], [cyan]{tau_max:+.4f}[/cyan]]"
    )


def print_rc_summary(rc: object) -> None:
    """Print residual confounding null-diagnostic summary (Fink §2.2).

    Values near zero indicate the propensity score model has absorbed the
    interannual confounding.  bias_ate_abd is the headline: how much trend
    the model estimates when the true trend is zero.
    """
    import math

    def _style(b: float) -> str:
        return "green" if abs(b) < 0.005 else ("yellow" if abs(b) < 0.02 else "red")

    bias_occ = getattr(rc, "bias_ate_occ", math.nan)
    bias_count = getattr(rc, "bias_ate_count", math.nan)
    bias_abd = getattr(rc, "bias_ate_abd", math.nan)
    n_ok = getattr(rc, "n_successful", 0)

    so = _style(bias_occ)
    sc = _style(bias_count)
    sa = _style(bias_abd)

    console.print(
        f"  bias_occ=[{so}]{bias_occ:+.4f}[/{so}]"
        f"  bias_count=[{sc}]{bias_count:+.4f}[/{sc}]"
        f"  bias_abd=[{sa}]{bias_abd:+.4f}[/{sa}]"
        f"  [dim]n={n_ok}[/dim]"
    )


def print_config_panel(cfg: object, run_ensemble: bool) -> None:
    """Print a configuration summary panel."""
    lines = [
        f"[dim]Species:[/dim]       {', '.join(cfg.species)}",  # type: ignore[attr-defined]
        f"[dim]Years:[/dim]         {cfg.years[0]}-{cfg.years[-1]}",  # type: ignore[attr-defined]
        f"[dim]PCA components:[/dim] {cfg.n_pca_components} per stat group",  # type: ignore[attr-defined]
        f"[dim]RF estimators:[/dim] {cfg.n_estimators} (nuisance)  {cfg.cf_n_estimators} (causal forest)",  # type: ignore[attr-defined]
        f"[dim]Parallelism:[/dim]   {cfg.n_workers} worker(s)  •  inner_n_jobs={cfg.inner_n_jobs}",  # type: ignore[attr-defined]
        f"[dim]Ensemble:[/dim]      {'enabled' if run_ensemble else 'disabled'}",
    ]
    console.print(Panel("\n".join(lines), title="[step]Configuration[/step]", expand=False))

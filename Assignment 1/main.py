"""
main.py  –  Full pipeline runner for Assignment 1: Portfolio VaR & ES
Run from the Assignment 1/ directory:  python main.py
"""

import sys
from pathlib import Path

# ensure the module directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

import module1_data        as m1
import module2_portfolio   as m2
import module3_var_es      as m3
import module4_rolling_var_es as m4
import module5_backtesting as m5
import module6_multiday_var as m6
import module7_stress_testing as m7


def run_all(force_rebuild: bool = False):
    print("\n" + "=" * 60)
    print("  Module 1 – Data")
    print("=" * 60)
    m1.load_or_build(force_rebuild=force_rebuild)

    print("\n" + "=" * 60)
    print("  Module 2 – Portfolio P&L")
    print("=" * 60)
    m2.build_portfolio(force_rebuild=force_rebuild)

    print("\n" + "=" * 60)
    print("  Module 3 – Static VaR & ES (5 methods) + period comparison")
    print("=" * 60)
    m3.estimate_var_es(force=force_rebuild)  # module3 uses 'force' kwarg

    print("\n" + "=" * 60)
    print("  Module 4 – Rolling VaR & ES")
    print("=" * 60)
    m4.compute_rolling_risk()

    print("\n" + "=" * 60)
    print("  Module 5 – Backtesting")
    print("=" * 60)
    m5.run_backtesting()

    print("\n" + "=" * 60)
    print("  Module 6 – Multi-Day VaR (HS vs. sqrt-of-time)")
    print("=" * 60)
    m6.run(force_rebuild=force_rebuild)

    print("\n" + "=" * 60)
    print("  Module 7 – Stress Testing")
    print("=" * 60)
    m7.run(force_rebuild=force_rebuild)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full QFRM Assignment 1 pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download and rebuild all data")
    args = parser.parse_args()
    run_all(force_rebuild=args.force)

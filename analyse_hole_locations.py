"""Utility checks for barrier hole detection.

This script verifies that the enhanced hole finding logic correctly labels
cells inside the U-shaped barrier of ``gridworlds.big_simple_grid``.
Running it will raise an exception if the regression ever reappears.
"""

EXPECTED = {(7, 7), (8, 7)}


def main() -> None:
    from structure_functions_ import get_hole_locations
    import gridworlds

    holes = set(get_hole_locations(gridworlds.big_simple_grid))
    missing = EXPECTED - holes
    if missing:
        raise SystemExit(
            f"Missing expected holes: {sorted(missing)}. Found holes: {sorted(holes)}"
        )
    print("Hole detection ok. Found:", sorted(holes))


if __name__ == "__main__":
    main()

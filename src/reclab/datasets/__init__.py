"""Dataset generation and loaders.

`synthetic` generates a reproducible, parameterized dataset for local
development, CI, and demos — used because the public benchmarks referenced in
docs/architecture/mvp-plan.md (MovieLens, Amazon Reviews) live on domains this
development sandbox can't reach. `loaders` implements real parsers for those
public datasets, correct against their published schemas, meant to be run on
a machine with normal internet access (see loaders.py module docstring for
exactly what's been verified vs. not).
"""

from reclab.datasets.synthetic import SyntheticConfig, generate_synthetic_dataset

__all__ = ["SyntheticConfig", "generate_synthetic_dataset"]

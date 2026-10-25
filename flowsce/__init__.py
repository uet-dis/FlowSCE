"""
FlowSCE — graph-consistent sparse counterfactual explanations for flow-level GNN intrusion detectors.

Every explainer (FlowSCE variants and all baselines) queries the SAME policy-consistent oracle
f̄_{π,k}(a'_k) = f_θ(T_{π,k}(a'_k − a_k))_k implemented by `QueryOracle`.

Modules (reference implementation, sequential)       Fast path
  config    configuration, dataset registry, utils      fast   batched / compressed evaluation, numerically
  data      loading, dedup group split, preprocessing          equivalent to the reference modules
  schema    feature schema, validator V_k, projection Π
  graph     graph-construction policy                  The package namespace re-exports the public API, so
  detector  pure-torch GAT detector                    `import flowsce as L; L.explain_query(...)` works.
  oracle    QueryOracle (modes A / B / C)              `L.POOL` (training-benign kNN pool used by the metrics)
  space     ξ coordinates, metric, domain box          reads/writes `flowsce.metrics.POOL`.
  explain   Algorithm 1: A0 / A / B / C / D
  metrics   assessment, ledger rows, CSR / SVR / quality
  checks    RQ1 reconstruction and gradient checks
  baselines CANDICE-style, DiCE-style, LBS-adapted, minimizers
  stats     detector summary, paired cluster bootstrap
  pipeline  one-call dataset preparation
"""
import sys as _sys, types as _types

from .config import *      # noqa: F401,F403
from .config import _np    # noqa: F401
from .data import *        # noqa: F401,F403
from .schema import *      # noqa: F401,F403
from .graph import *       # noqa: F401,F403
from .detector import *    # noqa: F401,F403
from .oracle import *      # noqa: F401,F403
from .space import *       # noqa: F401,F403
from .metrics import *     # noqa: F401,F403
from .explain import *     # noqa: F401,F403
from .checks import *      # noqa: F401,F403
from .baselines import *   # noqa: F401,F403
from .baselines import _final_row  # noqa: F401
from .stats import *       # noqa: F401,F403
from .pipeline import *    # noqa: F401,F403
from . import metrics as _metrics
from . import fast         # noqa: F401  (imports this namespace; defined after the reference API above)

globals().pop("POOL", None)  # the single source of truth is flowsce.metrics.POOL


class _Package(_types.ModuleType):
    """Forward `flowsce.POOL` to `flowsce.metrics.POOL` (module-level state read by assess() and LBS)."""
    @property
    def POOL(self):
        return _metrics.POOL

    @POOL.setter
    def POOL(self, value):
        _metrics.POOL = value


_sys.modules[__name__].__class__ = _Package

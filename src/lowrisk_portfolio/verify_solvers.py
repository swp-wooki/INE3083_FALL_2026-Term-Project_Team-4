"""Regenerate ratio/risk-budget certificates from the CURRENT observed-return inputs.

No historical certificate is read. Negative-numerator months are independently
checked by enumerating every capped-simplex vertex in a small compiled helper.
"""

from .paths import OUTPUT, ROOT

P = OUTPUT
OUT = OUTPUT
import itertools
import json
import math
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audit_helpers as ref
from .experiment import CAP, PRIMARY

# Exhaustive enumeration is independent of the production branch-and-bound.
VERTICES = r"""
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
int main(){
 int n,k;std::cin>>n>>k;
 std::vector<double>S(n*n),m(n),rows(n,0.0);for(auto &x:S)std::cin>>x;for(auto &x:m)std::cin>>x;
 std::vector<int>c(k),bestc(k);for(int i=0;i<k;i++)c[i]=i;
 double q=0.,a=0.,best=-1e300,maxerr=0.;unsigned long long count=0;
 auto reset=[&](){double old=q;q=0.;a=0.;std::fill(rows.begin(),rows.end(),0.);
   for(int j:c){a+=m[j];for(int i=0;i<n;i++)rows[i]+=S[i*n+j];}
   for(int j:c)q+=rows[j];if(count)maxerr=std::max(maxerr,std::abs(old-q));};
 reset();
 while(true){
   if(count%65536==0)reset();
   double score=a/std::sqrt(q);
   if(score>best){double aq=0.,am=0.;for(int i:c){am+=m[i];for(int j:c)aq+=S[i*n+j];}
     double exact=am/std::sqrt(aq);if(exact>best){best=exact;bestc=c;}}
   ++count;int pos=k-1;while(pos>=0 && c[pos]==n-k+pos)--pos;if(pos<0)break;
   for(int p=pos;p<k;p++){int j=c[p];q-=2*rows[j]-S[j*n+j];a-=m[j];for(int i=0;i<n;i++)rows[i]-=S[i*n+j];}
   ++c[pos];for(int p=pos+1;p<k;p++)c[p]=c[p-1]+1;
   for(int p=pos;p<k;p++){int j=c[p];q+=2*rows[j]+S[j*n+j];a+=m[j];for(int i=0;i<n;i++)rows[i]+=S[i*n+j];}
 }
 std::cout<<std::setprecision(17)<<best<<" "<<count<<" "<<maxerr;
 for(int j:bestc)std::cout<<" "<<j;std::cout<<"\n";
}
"""


def enumerator(directory):
    source = directory / "vertices.cpp"
    executable = directory / "vertices"
    source.write_text(VERTICES)
    subprocess.run(
        ["clang++", "-O3", "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )

    def solve(covariance, numerator, k):
        text = f"{len(numerator)} {k}\n" + " ".join(map(str, covariance.ravel())) + "\n"
        text += " ".join(map(str, numerator)) + "\n"
        result = subprocess.run(
            [str(executable)], input=text, text=True, capture_output=True, check=True
        )
        values = result.stdout.split()
        objective, count, error = float(values[0]), int(values[1]), float(values[2])
        if count != math.comb(len(numerator), k):
            raise RuntimeError("Incomplete independent vertex enumeration")
        return objective, count, error, list(map(int, values[3:]))

    # Small independent fixture checks the helper against direct Python sums.
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 8))
    S = A.T @ A
    mu = -rng.uniform(0.1, 2, 8)
    expected = max(
        mu[list(c)].sum() / np.sqrt(S[np.ix_(c, c)].sum())
        for c in itertools.combinations(range(8), 4)
    )
    if abs(solve(S, mu, 4)[0] - expected) > 1e-12:
        raise RuntimeError("Independent enumerator fixture failed")
    return solve


def run():
    start = time.monotonic()
    data, manifest = ref.inputs()
    if manifest.get("status") != "complete" or manifest.get("primary") != PRIMARY:
        raise RuntimeError("Run the observed-return experiment before solver verification")
    for name, sha in manifest["hashes"].items():
        if ref.digest(ROOT / name) != sha:
            raise RuntimeError("Stale experiment input: " + name)
    ratio = ref.module("ratio_solver").maximize_ratio
    risk = ref.module("erc").constrained_risk_budgeting
    rows = []
    negative_count = 0
    rf_column = list(data["factor_names"]).index("RF")
    with tempfile.TemporaryDirectory(prefix="ife_current_solver_cert_") as temporary:
        solve_vertices = None
        for i, X in enumerate(data["X"]):
            S = np.cov(X, rowvar=False)
            C = ref.module("cov_lw_identity").estimate(X)
            mu = ref.module("max_sharpe_bs").bayes_stein(X, C) - data["F"][i, -1, rf_column]
            record = {"month": str(data["months"][i]), "ids": data["ids"][i].tolist()}
            for name, covariance, numerator in [
                ("mdp", S, np.sqrt(S.diagonal())),
                ("max_sharpe_bs", C, mu),
            ]:
                w, info = ratio(covariance, numerator, cap=CAP, return_info=True)
                info.update(
                    objective=float(numerator @ w / np.sqrt(w @ covariance @ w)),
                    weights=w.tolist(),
                )
                if info["method"] == "vertex_branch_and_bound":
                    if solve_vertices is None:
                        solve_vertices = enumerator(Path(temporary))
                    print(
                        "Fresh exhaustive negative-ratio reference:",
                        record["month"],
                        flush=True,
                    )
                    objective, count, error, indices = solve_vertices(
                        covariance, numerator, round(1 / CAP)
                    )
                    difference = abs(info["objective"] - objective)
                    if difference > 1e-12:
                        raise RuntimeError("Negative-ratio objective disagrees with enumeration")
                    info.update(
                        exhaustive_reference_objective_difference=difference,
                        exhaustive_reference_vertices=count,
                        exhaustive_reference_quadratic_accumulation_error=error,
                        exhaustive_reference_indices=indices,
                    )
                    negative_count += 1
                elif info["method"] != "homogeneous_convex_qp":
                    raise RuntimeError(
                        "Unexpected ratio branch requires explicit independent verification"
                    )
                record[name] = info
            w, info = risk(S, cap=CAP, return_info=True)
            contributions = w * (S @ w)
            free = w < CAP - 1e-8
            info.update(
                weights=w.tolist(),
                uncapped_risk_contribution_spread=float(
                    np.ptp(contributions[free]) / contributions.sum()
                ),
                maximum_capped_relative_contribution=float(
                    np.max(contributions[~free] / contributions.sum(), initial=0)
                ),
            )
            record["erc"] = info
            rows.append(record)
            if (i + 1) % 50 == 0:
                print("Certified solver windows:", i + 1, flush=True)
    paths = [
        name
        for name in manifest["hashes"]
        if name.startswith("data/") or name.endswith("lowrisk_portfolio/data.py")
    ]
    paths += [str((OUT / "frozen_selection.csv").relative_to(ROOT))]
    code = [
        f"src/lowrisk_portfolio/optimizers/{name}.py"
        for name in ["mdp", "max_sharpe_bs", "erc", "ratio_solver", "cov_lw_identity"]
    ]
    code += [
        str(Path(__file__).relative_to(ROOT)),
        str((Path(__file__).parent / "audit_helpers.py").relative_to(ROOT)),
    ]
    result = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - start,
        primary_scenario=PRIMARY,
        scope="Fresh solves and fresh independent negative-ratio enumeration on current observed CRSP total returns; compare certified weights to saved holdings.",
        holdings_sha256=ref.digest(OUT / "holdings.csv.gz"),
        saved_holdings_verified=False,
        input_hashes={name: ref.digest(ROOT / name) for name in paths},
        code_hashes={name: ref.digest(ROOT / name) for name in code},
        portfolio_months_checked=3 * len(rows),
        negative_months_enumerated=negative_count,
        numerical_tolerances={
            "positive_ratio_relative_duality_gap": 1e-8,
            "negative_ratio_relative_objective_bound": 1e-10,
            "exhaustive_objective_agreement": 1e-12,
            "erc_budget": 1e-10,
            "erc_inner_relative_kkt": 1e-12,
        },
        assumptions=[
            "Finite positive-definite covariance; N=30, T=36 and cap=2/30.",
            "Float64 numerical certificates, not interval-arithmetic formal proofs.",
            "No earlier solver weights, input caches with stale hashes, or certificates reused.",
        ],
        rows=rows,
    )
    (OUT / "revision_solver_verification.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "portfolio_months_checked",
                    "negative_months_enumerated",
                    "elapsed_seconds",
                ]
            }
        )
    )


if __name__ == "__main__":
    run()

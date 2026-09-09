"""Verify the nine deliverables per design and print a table (also written to comparison/deliverable_check.json)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
LAM_ZE = json.load(open(HERE / "materials" / "material_models.json"))["lambda_ZE_data_nm"]
items = {
    "1 R/T/A with ITO": "prod/spectra_prod.csv",
    "2 R/T without ITO": "noito/spectra_noito.csv",
    "3 R/T/A at lambda_ZE": "prod/field_diagnostics.json",
    "4 complex Ex/Ey/Ez at lambda_ZE": "prod/fields_normalized.npz",
    "5 central xz Ez map": f"prod/fields_{LAM_ZE:.1f}nm.png",
    "6 ITO-plane xy Ez map": f"prod/fields_{LAM_ZE:.1f}nm.png",
    "7 Ux/Uy/Uz/Utot(z)": f"prod/Uz_{LAM_ZE:.1f}nm.csv",
    "8 multipole raw moments + diagnostics": "prod/field_diagnostics.json",
    "9 exact FDTD settings": "prod/run_meta.json",
}
res = {}
for d in ("final3", "final1", "final0", "final2"):
    res[d] = {k: (HERE / d / v).exists() for k, v in items.items()}
    print(d, {k.split()[0]: ("ok" if v else "MISSING") for k, v in res[d].items()})
json.dump(res, open(HERE / "comparison" / "deliverable_check.json", "w"), indent=1)

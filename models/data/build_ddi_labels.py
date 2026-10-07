"""Build real drug-drug-interaction (DDI) ground-truth labels for EquiPath.

Hetionet has no drug-drug-interaction edges of its own (its only compound-
compound relation is CrC, "resembles", based on structural similarity - not
interaction). This script brings in an external, real-world DDI dataset -
the Decagon polypharmacy side-effect dataset (Zitnik et al. 2018, derived
from the FDA's FAERS adverse-event reports via TWOSIDES/STITCH) - and links
it to Hetionet's DrugBank-ID compound nodes so DDI prediction has genuine
labels to train and evaluate against.

Steps:
  1. Download bio-decagon-combo.csv (drug A, side effect, drug B triples;
     drug IDs are STITCH flat compound IDs, i.e. "CID0" + zero-padded
     PubChem CID) if not already cached locally.
  2. Map each STITCH/PubChem CID to a DrugBank ID via PubChem's public
     PUG-REST xrefs/RegistryID endpoint (cached locally - this is an
     external network lookup, not something to redo on every run).
  3. Restrict to drugs that both (a) got a DrugBank ID and (b) are present
     as Compound nodes in Hetionet - only "usable" drugs, deduplicated
     drug pairs.
  4. Save the resulting positive-label DDI pairs and the drug universe to
     data/processed/ - negative pairs are generated at train time by
     models/prediction/ddi.py, not here.

Not every Decagon drug maps into Hetionet's DrugBank ID space (Decagon uses
STITCH/PubChem identifiers for a different drug cohort than Hetionet's
DrugBank-based compound set) - roughly half typically resolve. That's an
expected, honest limitation of linking two independently-built resources,
not a bug.
"""

import csv
import json
import re
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "decagon"
PROCESSED_DIR = ROOT / "data" / "processed"
HETIONET_NODES = ROOT / "data" / "raw" / "hetionet" / "nodes.tsv"

DECAGON_URL = "http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz"
COMBO_TARBALL = RAW_DIR / "bio-decagon-combo.tar.gz"
COMBO_CSV = RAW_DIR / "bio-decagon-combo.csv"
CID_MAP_CACHE = RAW_DIR / "cid_to_drugbank.json"

DRUGBANK_ID_RE = re.compile(r"^DB\d{5,6}$")
PUBCHEM_XREFS_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/xrefs/RegistryID/JSON"
)


def download_decagon() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if COMBO_CSV.exists():
        print(f"  Using cached {COMBO_CSV}")
        return

    print(f"  Downloading {DECAGON_URL} ...")
    urllib.request.urlretrieve(DECAGON_URL, COMBO_TARBALL)
    with tarfile.open(COMBO_TARBALL) as tar:
        tar.extractall(RAW_DIR, filter="data")
    print(f"  Extracted to {COMBO_CSV}")


def load_decagon_pairs() -> tuple[set[str], set[frozenset]]:
    """Return (unique STITCH CIDs, unique drug-pair frozensets) from the raw file."""
    cids: set[str] = set()
    pairs: set[frozenset] = set()
    with open(COMBO_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            cid1, cid2 = row[0], row[1]
            cids.add(cid1)
            cids.add(cid2)
            pairs.add(frozenset((cid1, cid2)))
    return cids, pairs


def stitch_to_pubchem_cid(stitch_id: str) -> int:
    """'CID000002173' -> 2173 (STITCH flat/stereo-unspecified CID -> PubChem CID)."""
    return int(stitch_id.replace("CID", ""))


def map_cids_to_drugbank(pubchem_cids: list[int]) -> dict[str, str]:
    """Query PubChem's RegistryID cross-references in batches, cached to disk."""
    if CID_MAP_CACHE.exists():
        print(f"  Using cached CID->DrugBank mapping: {CID_MAP_CACHE}")
        with open(CID_MAP_CACHE) as f:
            return json.load(f)

    print(f"  Querying PubChem for {len(pubchem_cids)} compounds (batched, rate-limited)...")
    cid_to_drugbank: dict[str, str] = {}
    batch_size = 100
    for i in range(0, len(pubchem_cids), batch_size):
        batch = pubchem_cids[i : i + batch_size]
        url = PUBCHEM_XREFS_URL.format(cids=",".join(map(str, batch)))
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            for info in data.get("InformationList", {}).get("Information", []):
                cid = info["CID"]
                registry_ids = info.get("RegistryID", [])
                drugbank_ids = [r for r in registry_ids if DRUGBANK_ID_RE.match(r)]
                if drugbank_ids:
                    cid_to_drugbank[str(cid)] = drugbank_ids[0]
        except Exception as e:
            print(f"    batch starting at {i} failed: {e}")
        time.sleep(0.3)  # be polite to PubChem's public endpoint

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(CID_MAP_CACHE, "w") as f:
        json.dump(cid_to_drugbank, f, indent=2)
    print(f"  Mapped {len(cid_to_drugbank)}/{len(pubchem_cids)} to a DrugBank ID, cached to {CID_MAP_CACHE}")
    return cid_to_drugbank


def load_hetionet_compound_ids() -> set[str]:
    compound_ids = set()
    with open(HETIONET_NODES) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3 and parts[2] == "Compound":
                compound_ids.add(parts[0].split("::")[1])  # "DB00945" from "Compound::DB00945"
    return compound_ids


def main() -> None:
    print("Step 1/4: Downloading Decagon polypharmacy dataset...")
    download_decagon()

    print("Step 2/4: Loading STITCH drug pairs...")
    stitch_cids, stitch_pairs = load_decagon_pairs()
    pubchem_cids = sorted({stitch_to_pubchem_cid(c) for c in stitch_cids})
    print(f"  {len(stitch_cids)} unique STITCH drugs, {len(stitch_pairs)} unique interacting pairs")

    print("Step 3/4: Mapping STITCH/PubChem CIDs to DrugBank IDs...")
    cid_to_drugbank = map_cids_to_drugbank(pubchem_cids)

    print("Step 4/4: Restricting to drugs present in Hetionet...")
    hetionet_compounds = load_hetionet_compound_ids()
    # stitch CID "CID000002173" -> plain pubchem cid string "2173" -> drugbank id
    stitch_to_drugbank = {}
    for stitch_id in stitch_cids:
        pubchem_cid = str(stitch_to_pubchem_cid(stitch_id))
        drugbank_id = cid_to_drugbank.get(pubchem_cid)
        if drugbank_id and drugbank_id in hetionet_compounds:
            stitch_to_drugbank[stitch_id] = drugbank_id

    usable_drugbank_ids = sorted(set(stitch_to_drugbank.values()))
    print(f"  {len(usable_drugbank_ids)} drugs usable (mapped to DrugBank AND present in Hetionet)")

    positive_pairs = set()
    for pair in stitch_pairs:
        stitch1, stitch2 = tuple(pair)
        db1 = stitch_to_drugbank.get(stitch1)
        db2 = stitch_to_drugbank.get(stitch2)
        if db1 and db2 and db1 != db2:
            positive_pairs.add(frozenset((db1, db2)))

    print(f"  {len(positive_pairs)} usable positive DDI pairs")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = PROCESSED_DIR / "ddi_labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["compound_1", "compound_2"])
        for pair in sorted(positive_pairs, key=lambda p: sorted(p)):
            d1, d2 = sorted(pair)
            writer.writerow([f"Compound::{d1}", f"Compound::{d2}"])
    print(f"Wrote {labels_path}")

    universe_path = PROCESSED_DIR / "ddi_drug_universe.csv"
    with open(universe_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["compound_id"])
        for drugbank_id in usable_drugbank_ids:
            writer.writerow([f"Compound::{drugbank_id}"])
    print(f"Wrote {universe_path}")


if __name__ == "__main__":
    main()

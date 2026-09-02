#!/usr/bin/env python3
"""
fetch_corpus.py — robust download of ~N recent open-access RCT PDFs.
Uses Europe PMC API to fetch recent medical papers (strictly 2025-2026).
Uses RCT-Reviewer's OWN SVM MODEL to verify if the paper is an RCT before downloading.
Prints detailed logs for every single paper evaluated.
"""

import argparse
import csv
import json
import time
import requests
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RCT_REPO = BASE / "RCT-Reviewer"
sys.path.insert(0, str(RCT_REPO))

# Headers for JSON API search
UA_JSON = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://europepmc.org/"
}

# Headers for PDF download
UA_PDF = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://europepmc.org/"
}

def download_pdf(pmcid, dest):
    pdf_url = f"https://europepmc.org/articles/PMC{pmcid}?pdf=render"
    try:
        resp = requests.get(pdf_url, headers=UA_PDF, timeout=60, allow_redirects=True)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('application/pdf'):
            dest.write_bytes(resp.content)
            return True
        else:
            return False
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "corpus"))
    ap.add_argument("--pmcid-list", default=None,
                    help="CSV with a pmcid column (e.g. corpus/metadata.csv): re-download "
                         "exactly these papers instead of querying Europe PMC, so an "
                         "existing corpus can be rebuilt 1:1 from its metadata")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.pmcid_list:
        with open(args.pmcid_list, newline="", encoding="utf-8") as f:
            # Only rows the classifier accepted were ever part of the corpus;
            # skipped (not-RCT) rows have no PDF to rebuild.
            rows = [r for r in csv.DictReader(f)
                    if r.get("pmcid") and r.get("is_rct") == "True"]
        ok = 0
        for r in rows:
            pmcid = r["pmcid"].replace("PMC", "")
            fname = f"PMC{pmcid}.pdf"
            dest = out / fname
            if dest.exists() and dest.stat().st_size > 10000:
                r["filename"], r["status"] = fname, "cached"
                ok += 1
                continue
            if download_pdf(pmcid, dest):
                r["filename"], r["status"], r["is_rct"] = fname, "downloaded", "True"
                ok += 1
            else:
                r["status"] = "download_failed"
                if dest.exists():
                    dest.unlink()
            time.sleep(0.5)
        with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pmcid", "title", "year",
                                              "model_score", "is_rct", "filename", "status"])
            w.writeheader()
            w.writerows(rows)
        print(f"DONE: {ok}/{len(rows)} corpus PDFs present in {out}")
        return

    print("Initializing RCT-Reviewer SVM Model...")
    from rct_reviewer.ml.rct_robot import RCTRobot
    new_robot = RCTRobot()
    print("Model loaded successfully!\n")

    print("Fetching open-access papers from Europe PMC (Strictly 2025-2026)...")
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    # Strict query: open access, has PDF, ONLY 2025 and 2026
    query = '(OPEN_ACCESS:y) AND (HAS_PDF:y) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)'
    
    page_size = 500
    cursor_mark = "*" # EPMC pagination token
    n_ok = 0
    meta_rows = []
    # cursorMark pagination can revisit records; dedupe by PMCID.
    seen_pmcids = set()
    page = 1

    while n_ok < args.target:
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "resultType": "core",
            "cursorMark": cursor_mark
        }
        
        try:
            resp = requests.get(url, params=params, headers=UA_JSON, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("resultList", {}).get("result", [])
            
            # Stop when the API stops advancing the cursor.
            if not hits:
                print("\nNo more records found from Europe PMC for 2025-2026.")
                break
                
            new_cursor_mark = data.get("nextCursorMark", cursor_mark)
            if new_cursor_mark == cursor_mark:
                print("\nReached the end of available Europe PMC results.")
                break
            cursor_mark = new_cursor_mark
            
            print(f"\n=========================================")
            print(f"Page {page}: Retrieved {len(hits)} candidates from API.")
            print(f"Verified RCTs collected so far: {n_ok} / {args.target}")
            print(f"=========================================")
            page += 1
            
            for hit in hits:
                if n_ok >= args.target:
                    break
                    
                pmcid = hit.get("pmcid", "").replace("PMC", "")
                if not pmcid:
                    continue
                if pmcid in seen_pmcids:
                    continue
                seen_pmcids.add(pmcid)

                title = hit.get("title", "No Title Available")
                abstract = hit.get("abstractText", "")
                year = hit.get("pubYear", "")

                if not title or not abstract:
                    print(f"\n[PMCID: PMC{pmcid}] Skipping: Missing title or abstract.")
                    continue

                short_title = (title[:70] + '...') if len(title) > 70 else title
                print(f"\n[PMCID: PMC{pmcid} | Year: {year}] Evaluating: '{short_title}'")

                # Dogfooding gate: only download papers RCT-Reviewer itself
                # classifies as RCTs (title + abstract input).
                try:
                    print(f"  -> Running SVM Model...")
                    pred = new_robot.predict(title, abstract)
                    is_rct = pred.get("is_rct", False)
                    score = pred.get("score", 0.0)
                    print(f"  -> Model Output: is_rct = {is_rct} | score = {score:.4f}")
                except Exception as e:
                    print(f"  -> Model Error: {e}. Skipping.")
                    is_rct = False
                    score = 0.0
                    
                if not is_rct:
                    print(f"  -> Action: Skipped (Model says NOT an RCT).")
                    meta_rows.append({
                        "pmcid": pmcid, "title": title, "year": year, 
                        "model_score": round(score, 4), "is_rct": False, 
                        "filename": "", "status": "skipped:not_rct"
                    })
                    continue # Skip if RCT-Reviewer says it's not an RCT!
                    
                fname = f"PMC{pmcid}.pdf"
                dest = out / fname
                
                if dest.exists() and dest.stat().st_size > 10000:
                    print(f"  -> Action: Already downloaded (Cached).")
                    status = "cached"
                    n_ok += 1
                else:
                    if dest.exists():
                        dest.unlink()
                        
                    print(f"  -> Action: Downloading PDF...")
                    if download_pdf(pmcid, dest):
                        print(f"  -> Status: Successfully downloaded.")
                        status = "downloaded"
                        n_ok += 1
                    else:
                        print(f"  -> Status: Download failed (server error or blocked).")
                        status = "download_failed"
                        if dest.exists():
                            dest.unlink()
                            
                meta_rows.append({
                    "pmcid": pmcid, "title": title, "year": year, 
                    "model_score": round(score, 4), "is_rct": True, 
                    "filename": fname if status in ("cached", "downloaded") else "", 
                    "status": status
                })
                
            time.sleep(0.5) # Politeness delay between pages
                
        except Exception as e:
            print(f"\nError searching Europe PMC: {e}")
            break

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmcid", "title", "year", "model_score", "is_rct", "filename", "status"])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"\n=========================================")
    print(f"DONE: {n_ok} verified RCT PDFs saved in {out}")
    print(f"Total papers evaluated by SVM: {len(seen_pmcids)}")
    print(f"=========================================")

if __name__ == "__main__":
    main()
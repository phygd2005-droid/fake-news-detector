"""
Claim verifier: finds evidence on the web and decides TRUE / FAKE / UNCERTAIN.

Steps for every claim:
 1. (optional) Google Fact Check API  -> existing fact-checks of the claim
 2. Web search (DuckDuckGo text + news) -> evidence snippets
 3. NLI model compares each snippet with the claim (supports / contradicts)
 4. Fact-check sites and trusted news sites get a higher weight
 5. Scores are added up -> final verdict
"""
import re
import time
import requests

NLI_MODEL = "cross-encoder/nli-distilroberta-base"   # ~330 MB, downloaded on first use

FACTCHECK_SITES = [
    "snopes.com", "politifact.com", "factcheck.org", "boomlive.in", "altnews.in",
    "newschecker.in", "factly.in", "vishvasnews.com", "thequint.com/news/webqoof",
    "reuters.com/fact-check", "apnews.com/hub/ap-fact-check", "fullfact.org",
    "factcrescendo.com", "logically.ai", "indiatoday.in/fact-check", "afp.com",
    "factchecker.in", "leadstories.com", "misbar.com", "dfrac.org", "newsmeter.in",
    "pibfactcheck", "pib.gov.in/factcheck", "fact-check", "factcheck", "fact_check",
]

TRUSTED_SITES = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "indianexpress.com", "ndtv.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "nasa.gov", "space.com", "cnn.com", "theguardian.com", "nytimes.com",
    "aljazeera.com", "pib.gov.in", "wikipedia.org", "britannica.com",
    "espncricinfo.com", "cricbuzz.com", "news18.com", "indiatoday.in",
    "economictimes.indiatimes.com", "scroll.in", "thewire.in", "washingtonpost.com",
    "bloomberg.com", "npr.org", "nature.com", "science.org", "who.int", "isro.gov.in",
    "bcci.tv", "iplt20.com", "firstpost.com", "deccanherald.com", "livemint.com",
]

# Strong words: a debunk on ANY site
DEBUNK_PHRASES = [
    "fake", "false", "hoax", "scam", "fraud", "phishing", "misleading", "myth", "untrue",
    "not true", "debunk", "baseless", "fabricated", "doctored", "morphed", "no truth",
    "misinformation", "unfounded", "not offering", "not giving", "not providing",
    "no such", "clickbait", "fraudalert", "not announced",
]
# Weaker words: only trusted as a debunk when the page is on a fact-check site
WEAK_DEBUNK_PHRASES = ["rumour", "rumor", "denies", "denied", "clarif", "no evidence", "beware"]
# Words showing a fact-check CONFIRMS the claim
CONFIRM_PHRASES = ["is true", "is real", "is correct", "is genuine", "claim is true",
                   "rated true", "true:", "verified true", "indeed true"]

STOP = set("""a an the of in on at to for and or but is are was were be been being has have
had with by from as that this it its his her their he she they them will would can could may
might about after before into over under again than then there here not no nor so such very
just also who whom which what when where why how all any some more most other our out up
down off only own same too s t don now said says say""".split())

MIN_RELEVANCE = 0.5       # snippet must share about 3 key words with the claim
MIN_TOTAL_EVIDENCE = 0.7  # below this we say UNCERTAIN


# ---------------------------------------------------------------- helpers
def keywords(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def relevance(claim, text):
    """0..1 - how many of the claim's key words appear in the text (capped at 6 words)."""
    ck = keywords(claim)
    if not ck:
        return 0.0
    hit = len(ck & keywords(text))
    return min(1.0, hit / min(len(ck), 6))


def has(url, domains):
    u = (url or "").lower()
    return any(d in u for d in domains)


def make_query(claim, max_words=12):
    words = [w for w in re.findall(r"[A-Za-z0-9']+", claim) if w.lower() not in STOP]
    return " ".join(words[:max_words])


def split_claims(text, max_claims=3):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= 300:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s for s in sents if len(s.split()) >= 6]
    return sents[:max_claims] or [text[:300]]


def rating_stance(rating):
    r = (rating or "").lower()
    if "half" in r or "partly" in r or "mixture" in r:
        return None
    if any(w in r for w in ["false", "fake", "incorrect", "misleading", "pants on fire",
                            "satire", "wrong", "not true", "unproven", "no evidence", "hoax"]):
        return "refute"
    if any(w in r for w in ["true", "correct", "accurate", "real"]):
        return "support"
    return None


# ---------------------------------------------------------------- NLI model
_tok = None
_mdl = None


def load_nli():
    global _tok, _mdl
    if _mdl is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        _mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _mdl.eval()
    return _tok, _mdl


_nli_ok = True


def nli_status():
    return "on" if _nli_ok else "off (lite mode)"


def nli(premise, hypothesis):
    """Returns (entailment_prob, contradiction_prob).
    If torch/transformers are not installed (cloud 'lite' version) it returns (0, 0)
    and the verdict relies on fact-check sites, debunk keywords and source agreement."""
    global _nli_ok
    if not _nli_ok:
        return 0.0, 0.0
    try:
        import torch
        tok, mdl = load_nli()
        inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            probs = torch.softmax(mdl(**inputs).logits, dim=-1)[0]
        lab = {mdl.config.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        return lab.get("entailment", 0.0), lab.get("contradiction", 0.0)
    except Exception:
        _nli_ok = False
        return 0.0, 0.0


# ---------------------------------------------------------------- online sources
def web_search(query, n=8, errors=None):
    out = []
    if errors is None:
        errors = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except Exception as e:
        errors.append(f"search library not installed: {e}")
        return out
    for kind in ("text", "news"):
        try:
            with DDGS() as d:
                rows = d.text(query, max_results=n) if kind == "text" else d.news(query, max_results=n)
                for r in rows:
                    out.append({"title": r.get("title", ""), "body": r.get("body", ""),
                                "url": r.get("href") or r.get("url", "")})
        except Exception as e:
            errors.append(f"{kind} search failed: {str(e)[:120]}")
    return out


def google_fact_check(query, key):
    out = []
    try:
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query[:200], "key": key, "languageCode": "en"},
            timeout=10).json()
        for c in r.get("claims", []):
            for rev in c.get("claimReview", []):
                out.append({
                    "claim_text": c.get("text", ""),
                    "rating": rev.get("textualRating", ""),
                    "publisher": rev.get("publisher", {}).get("name", ""),
                    "url": rev.get("url", ""),
                })
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- main logic
def verify_claim(claim, google_key=""):
    support = refute = 0.0
    evidence = []
    errors = []

    # 1) Google Fact Check API (optional)
    if google_key:
        for item in google_fact_check(make_query(claim), google_key):
            if relevance(claim, item["claim_text"]) < 0.5:
                continue
            st = rating_stance(item["rating"])
            if st in ("refute", "support"):
                if st == "refute":
                    refute += 5.0
                else:
                    support += 5.0
                evidence.append({"stance": "fact-check: " + item["rating"], "weight": 5.0,
                                 "title": f"{item['publisher']} - {item['claim_text'][:90]}",
                                 "url": item["url"]})

    # 2) Web search
    query = make_query(claim)
    results, seen = [], set()
    for i, q in enumerate((query, query + " fact check")):
        if i:
            time.sleep(1)                       # be gentle, avoids rate limits
        for r in web_search(q, errors=errors):
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)

    debug = {"query": query, "search_results": len(results), "relevant_used": 0,
             "support_score": 0.0, "refute_score": 0.0, "search_errors": errors[:3]}

    if not results and not evidence:
        msg = "Web search returned no results."
        if errors:
            msg += " Reason: " + errors[0]
        else:
            msg += " Check your internet connection and try again."
        return {"claim": claim, "verdict": "UNCERTAIN", "confidence": 0.0, "evidence": [],
                "reason": msg, "debug": debug}

    # 3) Compare every snippet with the claim
    used = 0
    for r in results:
        text = f"{r['title']}. {r['body']}"
        rel = relevance(claim, text)
        if rel < MIN_RELEVANCE:
            continue
        if used >= 15:
            break
        used += 1
        url = r["url"]
        is_fc = has(url, FACTCHECK_SITES)
        if is_fc:
            w = 3.0
        elif has(url, TRUSTED_SITES):
            w = 2.0
        else:
            w = 1.0
        low = text.lower()
        debunk = any(p in low for p in DEBUNK_PHRASES)
        confirm = any(p in low for p in CONFIRM_PHRASES)

        if is_fc:
            # A fact-check article about a viral claim is usually a debunk
            debunk_fc = debunk or any(p in low for p in WEAK_DEBUNK_PHRASES)
            if confirm and not debunk_fc:
                s, f, stance = w * rel, 0.0, "fact-check: true"
            elif debunk_fc:
                s, f, stance = 0.0, 1.5 * w * rel, "fact-check: false / fake"
            else:
                s, f, stance = 0.0, 0.5 * w * rel, "fact-check (likely false)"
        else:
            ent, con = nli(text, claim)
            s = w * ent * rel
            f = w * con * rel if con > 0.7 else 0.0
            if debunk:
                f += w * rel
                stance = "denies / debunks"
            elif ent > 0.4 and ent > con:
                stance = "supports"
            elif con > 0.7:
                stance = "contradicts"
            else:
                stance = "related"
                s += 0.35 * w * rel             # relevant source reports the same topic
        support += s
        refute += f
        evidence.append({"stance": stance, "weight": w, "title": r["title"] or url, "url": url})

    debug["relevant_used"] = used
    debug["nli_model"] = nli_status()
    debug["support_score"] = round(support, 2)
    debug["refute_score"] = round(refute, 2)

    # 4) Decide
    total = support + refute
    evidence.sort(key=lambda e: -e["weight"])
    if total < MIN_TOTAL_EVIDENCE:
        return {"claim": claim, "verdict": "UNCERTAIN", "confidence": 0.0, "evidence": evidence,
                "reason": f"Found {len(results)} web results but only {used} were relevant - "
                          "not enough evidence.", "debug": debug}
    conf = max(support, refute) / total
    if refute >= support * 1.3:
        verdict, reason = "FAKE", "Online sources deny or contradict this claim."
    elif support >= refute * 1.3:
        verdict, reason = "TRUE", "Online sources support this claim."
    else:
        verdict, reason = "UNCERTAIN", "Sources are mixed - cannot decide."
    return {"claim": claim, "verdict": verdict, "confidence": conf, "evidence": evidence,
            "reason": reason, "debug": debug}


def verify_text(text, google_key=""):
    claims = split_claims(text)
    per = [verify_claim(c, google_key) for c in claims]
    verdicts = [p["verdict"] for p in per]
    if "FAKE" in verdicts:
        overall = "FAKE"
    elif "TRUE" in verdicts:
        overall = "TRUE"
    else:
        overall = "UNCERTAIN"
    main = next((p for p in per if p["verdict"] == overall), per[0])
    all_evidence = [e for p in per for e in p["evidence"]]
    return {"verdict": overall, "confidence": main["confidence"], "reason": main["reason"],
            "evidence": all_evidence[:10], "claims": per, "debug": main.get("debug")}
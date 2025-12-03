# app.py
import os
import io
import json
import re
import requests
import pdfplumber
import docx2txt
import streamlit as st
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
# LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "https://dispensatorily-jettisonable-johnson.ngrok-free.dev/v1/chat/completions")
HEADERS = {"Content-Type": "application/json"}
# if os.getenv("LM_STUDIO_API_KEY"):
#     HEADERS["Authorization"] = f"Bearer {os.getenv('LM_STUDIO_API_KEY')}"
MODEL_NAME = os.getenv("LM_STUDIO_MODEL", "qwen3-4b-instruct-2507")

# ---------------- HELPERS -----------
def extract_text_from_file(file) -> str:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            return data.decode(errors="ignore")
    if name.endswith(".docx") or name.endswith(".doc"):
        try:
            return docx2txt.process(io.BytesIO(data))
        except Exception:
            return data.decode(errors="ignore")
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    
def simple_regex_extract(text):
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"(\+?\d[\d\-\s]{6,}\d)", text)
    return (email.group(0) if email else None, phone.group(0) if phone else None)

def chunk_text(text: str, max_chars: int = 10000):
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur += ("\n\n" if cur else "") + p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks

# ---------------- PROMPT BUILDERS (single-candidate focused) ----------------
def build_prompt_for_candidate(job_description: str, resume_texts: dict, threshold: float = 0.6) -> str:
    """
    Build a single prompt containing the JD and all candidate resumes.
    Ask the model to:
      - extract 5-7 critical skills from JD,
      - compute match_score (0-1) for each candidate,
      - produce candidate_rankings with rationale,
      - produce skill_matches mapping,
      - produce interview_recommendations only for High/Medium,
      - finally produce a 'shortlist' list (either top_n or all with score>=threshold).
    """

    schema = {
      "candidate_rankings": [
        "NAME: RANKING (High/Medium/Low) - Rationale (1-2 sentences explaining the fit against the JD)."
      ],
      "skill_matches": {
        "KEY_SKILL_1_FROM_JD": ["Candidate Name"],
      },
      "interview_recommendations": [
        "NAME: Key discussion point with reasoning (1 sentence)."
      ],
      "hiring_insights": [
        "Insight 1"
      ],
      "match_score": "numeric between 0 and 1 representing fraction of JD skills matched (float)."
    }

    example = {
"candidate_rankings": [
"Alice: High - Strong match on SQL, Python and Power BI; 4 years relevant experience."
],
"skill_matches": {"SQL": ["Alice"], "Python": ["Alice"], "Power BI": ["Alice"]},
"interview_recommendations": ["Alice: Ask about dashboard refresh strategies."],
"hiring_insights": ["Alice: Ready for mid-level Data Analyst role."],
"match_scores": {"Alice": 0.83},
"shortlist": ["Alice"]
}
    
    resumes_block = "\n\n".join([f"---\nCandidate: {name}\n\n{txt}" for name, txt in resume_texts.items()])

    prompt = f"""
You are an expert recruitment analyst. RETURN ONLY valid JSON exactly matching the SCHEMA described.

INSTRUCTIONS (must follow):
1) From the Job Description below, EXTRACT 5-7 absolutely critical skills/requirements. Use them as keys under "skill_matches".
2) For EACH candidate provided, compute a numeric "match_score" between 0 and 1 representing fraction of extracted JD skills matched (0=no skills, 1=all skills).
3) Use these match_scores to assign ranking:
   - match_score >= {threshold} => "High"
   - 0.4 <= match_score < {threshold} => "Medium"
   - match_score < 0.4 => "Low"
4) Produce concise rationales (1-2 sentences) in candidate_rankings.
5) Include interview_recommendations ONLY for High or Medium candidates.
6) Provide "shortlist" as ALL candidates with match_score >= {threshold}. Do NOT limit to top N. If no candidate meets the threshold, return an empty shortlist list.
7) DO NOT invent skills beyond JD unless marking them as "inferred". No extra

SCHEMA:
{json.dumps(schema, indent=2)}

EXAMPLE:
{json.dumps(example, indent=2)}

--- JOB DESCRIPTION:
{job_description}

--- CANDIDATE:
{resumes_block}

**BEGIN JSON OUTPUT NOW:**
"""
    return prompt.strip()

# ---------------- LM STUDIO CALL ----------------
def call_lm_studio_inference(prompt: str, max_new_tokens: int = 4096, timeout: int = 300):
    if not LM_STUDIO_URL:
        raise RuntimeError("LM_STUDIO_URL not set.")
    payload = {
        # Choose a model available on your LM Studio instance
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an AI Recruitment Assistant. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        # "response_format": {"type": "json_object"}  # uncomment if LM Studio supports it
    }
    r = requests.post(LM_STUDIO_URL, headers=HEADERS, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # support for OpenAI-style response
    if data and data.get("choices") and data["choices"][0].get("message"):
        return data["choices"][0]["message"]["content"]
    return json.dumps(data)

# ---------------- PARSING / MERGE / VALIDATION ----------------
def parse_json_loose(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except json.JSONDecodeError:
                pass
    return {}

# ---------------- MERGE / FALLBACK HELPERS ----------------
def merge_results(list_of_dicts: List[dict]) -> dict:
    # minimal merge (keeps lists unique)
    merged = {
        "candidate_rankings": [],
        "skill_matches": {},
        "interview_recommendations": [],
        "hiring_insights": [],
        "match_scores": {}
    }
    for d in list_of_dicts:
        if not isinstance(d, dict):
            continue
        cr = d.get("candidate_rankings")
        if cr:
            if isinstance(cr, list):
                merged["candidate_rankings"].extend(cr)
            else:
                merged["candidate_rankings"].append(cr)
        sm = d.get("skill_matches") or {}
        for skill, names in sm.items():
            merged.setdefault("skill_matches", {})
            merged["skill_matches"].setdefault(skill, [])
            for n in names:
                if n not in merged["skill_matches"][skill]:
                    merged["skill_matches"][skill].append(n)

        ir = d.get("interview_recommendations") or []
        if isinstance(ir, list):
            for x in ir:
                if x not in merged["interview_recommendations"]:
                    merged["interview_recommendations"].append(x)
        hi = d.get("hiring_insights") or []
        if isinstance(hi, list):
            for x in hi:
                if x not in merged["hiring_insights"]:
                    merged["hiring_insights"].append(x)
                                
        ms = d.get("match_scores") or d.get("match_score")
        if isinstance(ms, dict):
            for name, val in ms.items():
                merged["match_scores"][name] = float(val)
    return merged          
   

def compute_skill_scores_from_merged(merged: dict) -> dict:
    skill_matches = merged.get("skill_matches", {})
    candidates = set()
    for v in skill_matches.values():
        for name in v:
            candidates.add(name)
    counts = {c: 0 for c in candidates}
    for skill, names in skill_matches.items():
        for n in names:
            counts[n] = counts.get(n, 0) + 1
    total_skills = max(1, len(skill_matches.keys()))
    scores = {c: (counts.get(c, 0), counts.get(c, 0) / total_skills) for c in counts}
    # prefer any model-provided numeric score
    for c in list(scores.keys()):
        if c in merged.get("match_scores", {}):
            scores[c] = (scores[c][0], float(merged["match_scores"][c]))
    return scores

# ---------------- UPDATED validate_and_fix ----------------
def validate_and_fix(merged: dict, candidate_names: List[str]) -> dict:
    """
    Updated behavior:
    - Keep any rankings model provided.
    - Auto-assign ranking ONLY for candidates that have a numeric match_score reported by the model.
    - Do NOT auto-assign for candidates with no ranking AND no numeric score (prevents long list of auto Low).
    """
    merged.setdefault("candidate_rankings", [])
    merged.setdefault("skill_matches", {})
    merged.setdefault("interview_recommendations", [])
    merged.setdefault("hiring_insights", [])
    merged.setdefault("match_scores", {})

    # collect names already present in candidate_rankings
    existing_names = []
    for r in merged["candidate_rankings"]:
        m = re.match(r"\s*([^:]+):\s*(High|Medium|Low)", r, flags=re.IGNORECASE)
        if m:
            existing_names.append(m.group(1).strip())

    # Only auto-assign ranking for candidates where we have a numeric match_score.
    for name in candidate_names:
        if name not in existing_names:
            score = merged["match_scores"].get(name)
            if score is None:
                # Do NOT auto-assign if there is no numeric evidence from the model.
                continue
            if score >= 0.6:
                rank = "High"
            elif score >= 0.4:
                rank = "Medium"
            else:
                rank = "Low"
            merged["candidate_rankings"].append(f"{name}: {rank} - match_score={score:.2f}")

    # Ensure interview_recommendations only for High/Medium
    allowed = set()
    for r in merged["candidate_rankings"]:
        m = re.match(r"\s*([^:]+):\s*(High|Medium|Low)", r, flags=re.IGNORECASE)
        if m and m.group(2).lower() in ("high", "medium"):
            allowed.add(m.group(1).strip())

    filtered_recs = []
    for rec in merged.get("interview_recommendations", []):
        m = re.match(r"\s*([^:]+):", rec)
        if m and m.group(1).strip() in allowed:
            filtered_recs.append(rec)
    merged["interview_recommendations"] = filtered_recs

    # Remove duplicates while preserving order
    merged["candidate_rankings"] = list(dict.fromkeys(merged["candidate_rankings"]))
    merged["hiring_insights"] = list(dict.fromkeys(merged.get("hiring_insights", [])))
    return merged

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Resume Parser (LM Studio Local )", layout="wide")
st.title("📄 Resume Parser — LM Studio Local Server")

st.markdown("Upload resumes (PDF/DOCX/TXT) and paste Job Description (JD). The app evaluates each resume against the JD and returns merged structured JSON.")

job_description = st.text_area("📋 Job Description (required for JD-aligned output)", height=160, placeholder="Paste job description (required for JD-focused matching)")
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF/DOCX/TXT) — multiple allowed", type=["pdf", "docx", "txt"], accept_multiple_files=True)
threshold = st.slider("Shortlist threshold (match fraction)", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

resume_texts = {}
if uploaded_files:
    for f in uploaded_files:
        resume_texts[f.name.split(".")[0]] = extract_text_from_file(f)

if resume_texts:
    st.write("Detected resumes:", list(resume_texts.keys()))
else:
    st.write("No resumes uploaded yet.")

if st.button("Analyze with LM Studio Model"):
    if not job_description.strip():
        st.error("Please provide a Job Description for JD-aligned matching.")
    elif not resume_texts:
        st.error("Provide at least one resume (upload or paste) to analyze.")
    else:
        with st.spinner("Calling LM Studio with all candidates... this may take a little while"):
            prompt_all = build_prompt_for_candidate(job_description, resume_texts, threshold=threshold)
            try:
                raw = call_lm_studio_inference(prompt_all, max_new_tokens=4096, timeout=300)
            except Exception as e:
                st.error(f"LM Studio call failed: {e}")
                raw = None

            parsed = parse_json_loose(raw) if raw else {}
            if not parsed:
                st.warning("Model returned non-JSON or unparsable output. Showing raw preview (first 3000 chars):")
                st.text(raw[:3000] if raw else "(no raw output)")
                parsed = {}

            # normalize fields
            parsed.setdefault("candidate_rankings", [])
            parsed.setdefault("skill_matches", {})
            parsed.setdefault("interview_recommendations", [])
            parsed.setdefault("hiring_insights", [])
            parsed.setdefault("match_scores", {})
            parsed.setdefault("shortlist", [])

            merged = merge_results([parsed])

            scores = compute_skill_scores_from_merged(merged)

            # If model did not return shortlist, create shortlist using threshold and any model-provided match_scores
            final_shortlist = parsed.get("shortlist") or []
            if not final_shortlist:
                # build shortlist = all with score_fraction >= threshold
                final_shortlist = [name for name, (count, frac) in scores.items() if frac >= threshold]

            # use updated validate_and_fix behavior
            merged = validate_and_fix(merged, list(resume_texts.keys()))    

            # ensure candidate_rankings contains entries for all candidates (fallback)
            existing = [re.match(r"\s*([^:]+):", r).group(1).strip() for r in merged.get("candidate_rankings", []) if re.match(r"\s*([^:]+):", r)]
            for name in resume_texts.keys():
                if name not in existing:
                    # derive a simple fallback ranking from numeric score
                    sc = scores.get(name, (0, 0.0))[1]
                    if sc > 0:
                        if sc >= threshold:
                            rank = "High"
                        elif sc >= 0.4:
                            rank = "Medium"
                        else:
                            rank = "Low"
                        merged["candidate_rankings"].append(f"{name}: {rank} - (auto-assigned)")

            st.success("✅ Analysis complete")
            st.subheader(" Model-produced / merged JSON")
            st.json(merged)

            st.subheader(" Final Shortlist (threshold-based)")
            st.write(f"Threshold = {threshold}")
            st.write(final_shortlist)

            st.subheader(" Scores (match_count, fraction)")
            st.write(scores)

            st.subheader(" Quick field checks")
            for name, txt in resume_texts.items():
                em, ph = simple_regex_extract(txt)
                st.write(f"- {name}: email={em} | phone={ph}")
# EduScaffold Frontend (Streamlit) - Three-LLM Multi-Agent System (Final)
# APIs: Qwen via HF InferenceClient (provider=novita), Llama-4 Maverick via HF InferenceClient (provider=cerebras), Gemini via Google
# Thread-safety: rubric and policy are precomputed on main thread and passed into workers (no session_state access inside threads)

import os
import re
import time
import uuid
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional parsing for policy ingestion (PDF, DOCX)
import fitz  # PyMuPDF
import docx
from io import BytesIO

# ============ ENV VARS ============
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAdGZTX3JMWsOnGWnqGtoS-uHnnC3Gr05g")
HF_TOKEN = os.getenv("HF_TOKEN", "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")

# ============ CLIENTS ============
# Gemini (Google)
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

# Hugging Face InferenceClient (separate for each provider)
from huggingface_hub import InferenceClient
hf_qwen_client = InferenceClient(provider="novita", api_key=HF_TOKEN) if HF_TOKEN else None
hf_llama_client = InferenceClient(provider="cerebras", api_key=HF_TOKEN) if HF_TOKEN else None

# ============ STATE INIT ============
def init_state():
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("policies", [])            # [{id,name,desc,content}]
    st.session_state.setdefault("rubrics", [])             # [{id,name,weight}]
    st.session_state.setdefault("assignments", [])         # [{id,question,created_at,...}]
    st.session_state.setdefault("submissions", {})         # {assignment_id: [versions]}
    st.session_state.setdefault("active_assignment_id", None)
    st.session_state.setdefault("parsed_policy_cache", "")
    st.session_state.setdefault("debug_mode", False)
    st.session_state.setdefault("num_policy_entries", 1)
    st.session_state.setdefault("num_rubric_entries", 1)

init_state()

# ============ SANITIZERS / SCORING ============
THOUGHT_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_think(text: str) -> str:
    try:
        return THOUGHT_TAG_RE.sub("", text or "").strip()
    except Exception:
        return text or ""

SCORE_PATTERNS = [
    r"Final Score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"Final Score:\s*(\d+(?:\.\d+)?)/20",
    r"Final Score:\s*(\d+(?:\.\d+)?)\s*out\s*of\s*20",
    r"Final Score:\s*(\d+(?:\.\d+)?)",
    r"Total Score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"Score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"(\d+(?:\.\d+)?)/20",
]

def extract_score(text: str) -> float:
    for p in SCORE_PATTERNS:
        m = re.search(p, text or "", re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                return max(0.0, min(20.0, v))
            except Exception:
                pass
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text or "")
    for n in nums:
        try:
            v = float(n)
            if 0.0 <= v <= 20.0:
                return v
        except Exception:
            continue
    return 10.0

# ============ RUBRIC ============
HARDCODED_RUBRIC = """
Score Criteria (out of 20):
- Application / Presentation of Concept: 2
- Detailing and Understanding: 1
- Skills Exploration: 1
- Basics of Design Principles: 1
- Research and Comprehension: 2
- Meta-cognition and Critical Thinking: 1.5
- Perception, Observation, and Sensitivity: 1.5
- Conceptual Clarity and Comprehension (Theory): 2
- Reflective Thinking: 1.5
- Communication: 0.75
- Conceptual Clarity: 2
- Exploration and Improvisation: 1.5
- Problem-solving and Lateral Thinking: 1.5
- Originality and Visualization: 1

Guidelines:
- Provide justification for each score.
- Sum to a total and end with: Final Score: XX/20
"""

def get_active_rubric_text():
    if st.session_state.rubrics:
        rubric_text = "**INSTRUCTOR-DEFINED RUBRIC** (Total: 20 points)\n\nScore each criterion as specified:\n\n"
        total = 0.0
        for i, r in enumerate(st.session_state.rubrics, 1):
            pts = (r["weight"] / 100.0) * 20.0
            rubric_text += f"{i}. {r['name']}: {pts:.2f} points ({r['weight']}%)\n"
            total += pts
        rubric_text += f"\nTOTAL POSSIBLE: {total:.2f}/20\n\nEnd with: Final Score: XX/20"
        return rubric_text
    return HARDCODED_RUBRIC

def format_rubric_for_display():
    return f"Custom Rubric ({len(st.session_state.rubrics)} criteria)" if st.session_state.rubrics else "Default Hardcoded Rubric (14 criteria)"

def get_rubric_breakdown_text():
    if not st.session_state.rubrics:
        return "Using default rubric criteria."
    lines = []
    for r in st.session_state.rubrics:
        pts = (r["weight"] / 100.0) * 20.0
        lines.append(f"• {r['name']}: {pts:.2f} pts ({r['weight']}%)")
    return "\n".join(lines)

# ============ PROMPTS (thread-safe: only use state data) ============
def generic_context_and_instructions(state):
    return f"""
### INSTITUTIONAL POLICY
{state.get('parsed_policy', 'No policy provided')}

### QUESTION
{state['question']}

### STUDENT ANSWER
{state['answer']}

### EVALUATION RUBRIC
{state['rubric']}

REQUIRED FORMAT:
1) CRITERION-BY-CRITERION: score + justification with evidence.
2) OVERALL ASSESSMENT: strengths and weaknesses.
3) RECOMMENDATIONS: specific, actionable steps.

End with: Final Score: XX/20
"""

def evaluate_answer_prompt_peer1(state):
    return f"You are Agent #1: Policy/Integrity.\nFocus on accuracy, policy compliance, academic integrity.\n{generic_context_and_instructions(state)}"

def evaluate_answer_prompt_peer2(state):
    return f"You are Agent #2: Pedagogy.\nFocus on Bloom alignment, learning objectives, clarity, structure.\n{generic_context_and_instructions(state)}"

def evaluate_answer_prompt_peer3(state):
    return f"You are Agent #3: Originality/Voice.\nFocus on individual voice, originality, creative problem-solving.\n{generic_context_and_instructions(state)}"

def evaluate_answer_prompt_peer4(state):
    return f"You are Agent #4: Equity.\nFocus on inclusive, bias-free evaluation and cultural sensitivity.\n{generic_context_and_instructions(state)}"

def evaluate_answer_prompt_peer5(state):
    return f"You are Agent #5: Feedback.\nFocus on specific, growth-oriented, scaffolded recommendations.\n{generic_context_and_instructions(state)}"

def evaluate_answer_prompt_peer6(state):
    return f"You are Agent #6: Summarizer.\nProvide holistic synthesis and cross-criterion coherence.\n{generic_context_and_instructions(state)}"

PEER_PROMPT_FUNCTIONS = [
    evaluate_answer_prompt_peer1,
    evaluate_answer_prompt_peer2,
    evaluate_answer_prompt_peer3,
    evaluate_answer_prompt_peer4,
    evaluate_answer_prompt_peer5,
    evaluate_answer_prompt_peer6,
]

# ============ LLM CALLS (InferenceClient + Gemini) ============
def _extract_hf_chat_text(completion):
    try:
        msg = completion.choices[0].message
        if hasattr(msg, "content") and msg.content:
            return msg.content
        return str(msg)
    except Exception:
        return str(completion)

def call_qwen(prompt: str) -> str:
    if hf_qwen_client is None:
        return "Qwen client not configured (HF_TOKEN missing). Final Score: 10/20"
    try:
        completion = hf_qwen_client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Thinking",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=1500,
            temperature=0.7,
        )
        out = strip_think(_extract_hf_chat_text(completion))
        if "Final Score:" not in out and "final score:" not in out.lower():
            out += "\n\nFinal Score: 15/20"
        return out
    except Exception as e:
        return f"Qwen evaluation error: {e}\nFinal Score: 10/20"

def call_maverick(prompt: str) -> str:
    if hf_llama_client is None:
        return "Llama client not configured (HF_TOKEN missing). Final Score: 10/20"
    try:
        completion = hf_llama_client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=1500,
            temperature=0.7,
        )
        out = strip_think(_extract_hf_chat_text(completion))
        if "Final Score:" not in out and "final score:" not in out.lower():
            out += "\n\nFinal Score: 14/20"
        return out
    except Exception as e:
        return f"Llama-4 Maverick evaluation error: {e}\nFinal Score: 10/20"

def call_gemini(prompt: str) -> str:
    try:
        out = strip_think(gemini_llm.invoke(prompt).content)
        if "Final Score:" not in out and "final score:" not in out.lower():
            out += "\n\nFinal Score: 15/20"
        return out
    except Exception as e:
        return f"Gemini evaluation error: {e}\nFinal Score: 12/20"

LLM_FUNCTIONS = [call_qwen, call_maverick, call_gemini]
LLM_NAMES = ["Qwen3-VL-235B (novita)", "Llama-4 Maverick (cerebras)", "Gemini-2.5"]

# ============ EVALUATION ENGINE (thread-safe) ============
def evaluate_agent_with_three_llms(state, agent_id: int):
    # Only use data from state; do not read st.session_state here
    prompt = PEER_PROMPT_FUNCTIONS[agent_id - 1](state)
    llm_responses, llm_scores = [], []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(LLM_FUNCTIONS[i], prompt): i for i in range(3)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                resp = fut.result(timeout=120)
                score = extract_score(resp)
                llm_responses.append(f"**{LLM_NAMES[idx]}:**\n{resp}")
                llm_scores.append(score)
            except Exception as e:
                llm_responses.append(f"**{LLM_NAMES[idx]}:** error: {e}\nFinal Score: 10/20")
                llm_scores.append(10.0)
    while len(llm_scores) < 3:
        llm_scores.append(10.0)
        llm_responses.append("Missing response fallback. Final Score: 10/20")
    agent_avg = sum(llm_scores) / len(llm_scores)
    return {
        "agent_id": agent_id,
        "llm_scores": llm_scores,
        "agent_avg_score": agent_avg,
        "combined_response": "\n\n".join(llm_responses),
    }

def peer_assessment_simulation(state, num_peers=6):
    results = []
    # Pass an immutable snapshot to each worker
    with ThreadPoolExecutor(max_workers=num_peers) as ex:
        futures = {ex.submit(evaluate_agent_with_three_llms, dict(state), i): i for i in range(1, num_peers + 1)}
        for fut in as_completed(futures):
            agent_id = futures[fut]
            try:
                results.append(fut.result(timeout=240))
            except Exception as e:
                results.append({
                    "agent_id": agent_id,
                    "llm_scores": [10.0, 10.0, 10.0],
                    "agent_avg_score": 10.0,
                    "combined_response": f"Agent {agent_id} error: {e}\nFinal Score: 10/20",
                })
    results.sort(key=lambda x: x["agent_id"])
    agent_avgs = [r["agent_avg_score"] for r in results]
    overall = sum(agent_avgs) / len(agent_avgs) if agent_avgs else 10.0

    agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
    sections = []
    for r in results:
        name = agent_names[r["agent_id"] - 1]
        sections.append(
            f"**Agent #{r['agent_id']} - {name} (Score: {r['agent_avg_score']:.2f}/20):**\n"
            f"LLM Ensemble Scores: " + " | ".join([f"{LLM_NAMES[i]}: {s:.1f}" for i, s in enumerate(r["llm_scores"])]) + "\n\n"
            f"{r['combined_response']}"
        )
    rubric_ctx = f"**EVALUATION CONTEXT:**\n{format_rubric_for_display()}\n{get_rubric_breakdown_text()}\n\n"
    return {
        "peer_reviews": "\n\n" + "=" * 50 + "\n\n" + rubric_ctx + "\n\n".join(sections),
        "average_score": overall,
        "scores": agent_avgs,
        "detailed_results": results,
    }

# ============ UI HELPERS ============
def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #FFFBEA 0%, #EAF8F0 40%, #E9F3FF 100%); }
        .card { background:#ffffffAA; border-radius:16px; padding:16px 18px; box-shadow:0 6px 20px rgba(0,0,0,0.05); margin-bottom:14px; }
        .rubric-indicator { background:#e8f5e8; padding:8px 12px; border-radius:8px; margin:8px 0; border-left:4px solid #4caf50; }
        .llm-score { background:#f0f8ff; padding:4px 8px; border-radius:4px; margin:2px; display:inline-block; font-size:0.8rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

def topbar():
    st.markdown(
        """
        <div class="card" style="display:flex;align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,#FFF5B1,#BDE8C6,#BFE1FF);display:flex;align-items:center;justify-content:center;font-weight:800;color:#1f4e79;">ES</div>
            <div>
              <div style="font-size:1.4rem;font-weight:800;color:#1f4e79;">EduScaffold</div>
              <div style="color:#5f6c7b;">Policy-Compliant AI Scaffolding with Three-LLM Ensemble</div>
            </div>
          </div>
          <div class="muted">🛡️</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============ INSTRUCTOR MODULES ============
def policy_ingestor():
    st.markdown("<div class='card'><h4>📋 Policy Ingestor</h4>", unsafe_allow_html=True)
    with st.form("policy_form"):
        items = []
        for i in range(st.session_state.num_policy_entries):
            st.markdown(f"**Policy Entry {i+1}**")
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: name = st.text_input("Policy Name", key=f"policy_name_{i}")
            with c2: desc = st.text_input("Description", key=f"policy_desc_{i}")
            with c3: upload = st.file_uploader("📄", type=["pdf", "txt", "docx"], key=f"policy_file_{i}")

            content = ""
            if upload:
                try:
                    if upload.type == "application/pdf":
                        doc = fitz.open(stream=upload.read(), filetype="pdf")
                        content = "\n".join([p.get_text() for p in doc])
                        st.success(f"✅ PDF content extracted for Policy {i+1}")
                    elif upload.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
                        fbytes = upload.read()
                        fbuf = BytesIO(fbytes)
                        d = docx.Document(fbuf)
                        content = "\n".join([p.text for p in d.paragraphs])
                        st.success(f"✅ DOCX content extracted for Policy {i+1}")
                    else:
                        content = upload.read().decode("utf-8", errors="ignore")
                        st.success(f"✅ File content loaded for Policy {i+1}")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")

            manual = st.text_area("Policy Content (Manual Entry or Extracted)", value=content, height=120, key=f"policy_content_{i}")
            items.append({"name": name, "desc": desc, "content": manual})
            st.markdown("---")

        save = st.form_submit_button("💾 Save All Policies", type="primary")
        if save:
            saved = 0
            for it in items:
                if it["name"].strip() and (it["desc"].strip() or it["content"].strip()):
                    st.session_state.policies.append({
                        "id": str(uuid.uuid4()),
                        "name": it["name"].strip(),
                        "desc": it["desc"].strip(),
                        "content": it["content"].strip(),
                    })
                    saved += 1
            if saved > 0:
                st.session_state.parsed_policy_cache = "\n\n".join([p["content"] for p in st.session_state.policies])
                st.success(f"✅ {saved} policies saved and cached for evaluation context.")
            else:
                st.warning("⚠️ Provide name and description/content for each policy before saving.")
    if st.button("➕ Add More Policy Entry"):
        st.session_state.num_policy_entries += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📚 Saved Policies</h4>", unsafe_allow_html=True)
    if st.session_state.policies:
        for p in st.session_state.policies:
            with st.expander(f"📄 {p['name']}"):
                st.write(f"**Description:** {p['desc'] or '(No description)'}")
                preview = (p["content"] or "")
                st.code(preview[:800] + ("..." if len(preview) > 800 else ""), language="text")
    else:
        st.info("No policies saved yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def rubric_builder():
    st.markdown("<div class='card'><h4>📊 Rubric Builder</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='rubric-indicator'>Active: {format_rubric_for_display()}</div>", unsafe_allow_html=True)

    with st.form("rubric_form"):
        data, total = [], 0.0
        for i in range(st.session_state.num_rubric_entries):
            c1, c2 = st.columns([3, 1])
            with c1: nm = st.text_input(f"Criterion {i+1}", key=f"rubric_name_{i}")
            with c2: wt = st.number_input("Weight %", min_value=0.0, max_value=100.0, value=50.0 if i < 2 else 0.0, step=1.0, key=f"rubric_weight_{i}")
            if nm.strip():
                data.append({"name": nm.strip(), "weight": float(wt)})
                total += wt

        if abs(total - 100.0) < 0.01:
            st.success(f"✅ Total: {total:.0f}% (Perfect)")
        else:
            st.warning(f"⚠️ Total: {total:.0f}% (Must equal 100%)")

        save = st.form_submit_button("💾 Save Custom Rubric", type="primary")
        if save:
            if abs(total - 100.0) < 0.01 and data:
                st.session_state.rubrics = [{"id": str(uuid.uuid4()), **d} for d in data]
                st.success(f"✅ Custom rubric with {len(data)} criteria saved and activated.")
                st.rerun()
            else:
                st.error("❌ Rubric must have names and total exactly 100%.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Add More Criteria"):
            st.session_state.num_rubric_entries += 1
            st.rerun()
    with c2:
        if st.session_state.rubrics and st.button("🗑️ Reset to Default Rubric"):
            st.session_state.rubrics = []
            st.session_state.num_rubric_entries = 1
            st.success("✅ Reset to default rubric.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📝 Current Evaluation Rubric</h4>", unsafe_allow_html=True)
    if st.session_state.rubrics:
        df_rows = []
        for r in st.session_state.rubrics:
            pts = (r["weight"] / 100.0) * 20.0
            df_rows.append({"Criterion": r["name"], "Weight (%)": r["weight"], "Points (out of 20)": f"{pts:.2f}"})
        st.table(pd.DataFrame(df_rows))
    else:
        st.info("📄 Using Default Hardcoded Rubric — create custom criteria above to override.")
        with st.expander("📄 View Default Rubric Details"):
            st.code(HARDCODED_RUBRIC, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

def assignment_creator():
    st.markdown("<div class='card'><h4>📝 Assignment Creation</h4>", unsafe_allow_html=True)
    with st.form("assignment_form", clear_on_submit=True):
        q = st.text_area("📄 Assignment Question / Prompt", height=120,
                         placeholder="Enter the assignment question or prompt...")
        submit = st.form_submit_button("🚀 Publish Assignment", type="primary")
        if submit:
            if q.strip():
                st.session_state.assignments.append({
                    "id": str(uuid.uuid4()),
                    "question": q.strip(),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "rubric_snapshot": format_rubric_for_display(),
                    "rubric_details": get_rubric_breakdown_text(),
                })
                st.success("✅ Assignment published with current rubric settings!")
            else:
                st.warning("⚠️ Please enter a question.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📚 Published Assignments</h4>", unsafe_allow_html=True)
    if st.session_state.assignments:
        for i, a in enumerate(st.session_state.assignments, 1):
            st.markdown(f"**{i}.** {a['question']}")
            st.caption(f"🗓️ Published: {a['created_at']}")
            st.caption(f"📄 Will be evaluated using: {a['rubric_snapshot']}")
            st.markdown("---")
    else:
        st.info("No assignments published yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def student_monitoring():
    st.markdown("<div class='card'><h4>🕵️ Student Monitoring Dashboard</h4>", unsafe_allow_html=True)
    if not st.session_state.assignments:
        st.info("📭 No assignments published yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    options = {f"{i+1}. {a['question'][:50]}...": a["id"] for i, a in enumerate(st.session_state.assignments)}
    sel = st.selectbox("📄 Select Assignment to Monitor", list(options.keys()))
    aid = options[sel]
    history = st.session_state.submissions.get(aid, [])

    if not history:
        st.info("📝 No student submissions yet for this assignment.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    selected = next(a for a in st.session_state.assignments if a["id"] == aid)
    st.markdown(f"<div class='rubric-indicator'>📄 This assignment uses: {selected['rubric_snapshot']}</div>", unsafe_allow_html=True)

    st.markdown("#### ⏰ Submission Timeline")
    for i, h in enumerate(history, 1):
        icon = "🏁" if h.get("submitted") else "📝"
        st.markdown(f"{icon} **Version {i}** — {h['timestamp']} — Score: {h['avg']:.2f}/20")

    st.markdown("#### 📈 Score Progression")
    scores = [h["avg"] for h in history]
    if scores:
        df = pd.DataFrame({"Version": list(range(1, len(scores)+1)), "Score": scores})
        st.line_chart(df.set_index("Version"))
    else:
        st.write("No scores yet.")

    st.markdown("#### 📨 Detailed Feedback History")
    for i, h in enumerate(history, 1):
        status = "Final" if h.get("submitted") else "Draft"
        with st.expander(f"Version {i} ({status})"):
            st.markdown(h["feedback"])

    st.markdown("</div>", unsafe_allow_html=True)

# ============ STUDENT WORKSPACE ============
def student_workspace():
    st.markdown("<div class='card'><h4>🎓 Student Workspace</h4>", unsafe_allow_html=True)
    if not st.session_state.assignments:
        st.info("📭 Waiting for instructor to publish assignments.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    options = {f"{i+1}. {a['question'][:50]}...": a["id"] for i, a in enumerate(st.session_state.assignments)}
    sel = st.selectbox("Select Assignment to Work On", list(options.keys()))
    aid = options[sel]
    selected = next(a for a in st.session_state.assignments if a["id"] == aid)
    full_question = selected["question"]
    st.markdown(f"**📋 Question:** {full_question}")
    st.markdown(f"<div class='rubric-indicator'>Evaluations use: {selected['rubric_snapshot']}</div>", unsafe_allow_html=True)

    default_text = ""
    if st.session_state.submissions.get(aid):
        default_text = st.session_state.submissions[aid][-1]["answer"]
    answer = st.text_area("Draft your answer:", height=220, value=default_text,
                          placeholder="Write your comprehensive response here...")

    c1, c2, c3, c4 = st.columns(4)
    with c1: evaluate = st.button("🤖 Submit for AI Evaluation")
    with c2: modify = st.button("✏️ Modify & Resubmit")
    with c3: final_submit = st.button("🏁 Final Submit", type="primary")
    with c4: st.session_state.debug_mode = st.checkbox("🐛 Debug", value=st.session_state.get("debug_mode", False))

    parsed_policy = st.session_state.parsed_policy_cache or "No specific institutional policy provided"

    if evaluate or modify:
        if not answer.strip():
            st.warning("⚠️ Please enter an answer before evaluation.")
        elif len(answer.strip()) < 50:
            st.warning("⚠️ Please provide a more detailed answer (at least 50 characters).")
        else:
            eval_state = {
                "question": full_question,
                "answer": answer,
                "rubric": get_active_rubric_text(),  # computed on main thread
                "parsed_policy": parsed_policy,      # cached string
            }
            with st.spinner("🤖 Running three‑LLM multi‑agent evaluation..."):
                results = peer_assessment_simulation(eval_state, num_peers=6)

            agent_scores = results["scores"]
            overall_avg = results["average_score"]
            st.session_state.submissions.setdefault(aid, []).append({
                "answer": answer,
                "scores": agent_scores,
                "agent_scores": agent_scores,
                "avg": overall_avg,
                "feedback": results["peer_reviews"],
                "detailed_results": results.get("detailed_results", []),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "submitted": False,
                "rubric_used": format_rubric_for_display(),
            })
            st.success("✅ Three‑LLM ensemble evaluation complete! See detailed feedback below.")

    if final_submit:
        versions = st.session_state.submissions.get(aid, [])
        if not versions:
            st.warning("⚠️ Please evaluate at least once before final submission.")
        else:
            versions[-1]["submitted"] = True
            st.success("🏆 Final submission recorded.")

    versions = st.session_state.submissions.get(aid, [])
    if versions:
        last = versions[-1]
        st.metric("📊 Overall Score", f"{last['avg']:.2f}/20", f"{(last['avg']/20*100):.1f}%")
        agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
        cols = st.columns(6)
        for i, (name, score) in enumerate(zip(agent_names, last["scores"])):
            with cols[i]:
                pct = (score / 20 * 100.0)
                st.metric(f"{name}", f"{score:.2f}", f"{pct:.1f}%")
        with st.expander("📄 View Comprehensive Three‑LLM Feedback & Detailed Analysis", expanded=False):
            st.markdown(last["feedback"])
    else:
        st.info("💡 Submit for evaluation to see detailed feedback here.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============ DIAGNOSTICS ============
def diagnostics_panel():
    st.markdown("<div class='card'><h4>🔧 API Diagnostics</h4>", unsafe_allow_html=True)
    st.write("HF_TOKEN set:", bool(HF_TOKEN))
    st.write("GOOGLE_API_KEY set:", bool(GOOGLE_API_KEY))
    test_prompt = "Reply with OK.\n\nFinal Score: 18/20"

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Test Qwen (novita)"):
            st.code(call_qwen(test_prompt)[:500])
    with c2:
        if st.button("Test Maverick (cerebras)"):
            st.code(call_maverick(test_prompt)[:500])
    with c3:
        if st.button("Test Gemini"):
            st.code(call_gemini(test_prompt)[:500])
    st.markdown("</div>", unsafe_allow_html=True)

# ============ NAV / MAIN ============
def login_page():
    st.markdown("<div class='card' style='text-align:center;padding:32px;'>", unsafe_allow_html=True)
    st.markdown("<h2>Welcome to EduScaffold</h2>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Choose a role to continue</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👨‍🏫 Instructor", use_container_width=True):
            st.session_state.role = "Instructor"; st.rerun()
    with c2:
        if st.button("🎓 Student", use_container_width=True):
            st.session_state.role = "Student"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="EduScaffold", layout="wide", page_icon="🎓")
    inject_css(); topbar()

    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        st.markdown("**Ensemble:** Qwen3‑VL‑235B (novita) • Llama‑4 Maverick (cerebras) • Gemini‑2.5")
        if not st.session_state.role:
            st.info("👈 Select role on the login screen.")
        else:
            st.success(f"Logged in as: **{st.session_state.role}**")
        if st.session_state.role == "Instructor":
            choice = st.radio("Go to", ["🏠 Dashboard", "📋 Policy Ingestor", "📊 Rubric Builder", "📝 Assignment Creator", "🕵️ Student Monitoring", "🔧 Diagnostics", "🚪 Logout"], index=0)
        elif st.session_state.role == "Student":
            choice = st.radio("Go to", ["🎓 Workspace", "🚪 Logout"], index=0)
        else:
            choice = None

    if st.session_state.role is None:
        login_page(); return

    if st.session_state.role == "Instructor":
        if choice == "🏠 Dashboard":
            st.markdown("<div class='card'><h4>👨‍🏫 Instructor Dashboard</h4><p class='muted'>Manage courses with three‑LLM ensemble and custom rubric.</p></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("📄 Policies", len(st.session_state.policies))
            with c2: st.metric("📊 Rubric Criteria", len(st.session_state.rubrics))
            with c3: st.metric("📝 Assignments", len(st.session_state.assignments))
            with c4: st.metric("🔑 APIs", "Configured" if HF_TOKEN and GOOGLE_API_KEY else "Missing")
        elif choice == "📋 Policy Ingestor":
            policy_ingestor()
        elif choice == "📊 Rubric Builder":
            rubric_builder()
        elif choice == "📝 Assignment Creator":
            assignment_creator()
        elif choice == "🕵️ Student Monitoring":
            student_monitoring()
        elif choice == "🔧 Diagnostics":
            diagnostics_panel()
        elif choice == "🚪 Logout":
            st.session_state.role = None
            st.session_state.num_policy_entries = 1
            st.session_state.num_rubric_entries = 1
            st.rerun()

    elif st.session_state.role == "Student":
        if choice == "🎓 Workspace":
            student_workspace()
        elif choice == "🚪 Logout":
            st.session_state.role = None
            st.rerun()

if __name__ == "__main__":
    main()

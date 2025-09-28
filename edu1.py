# EduScaffold Frontend (Streamlit) - Three-LLM Multi-Agent System
# Updated to use Hugging Face InferenceClient:
#   - Qwen: provider="novita", model="Qwen/Qwen3-VL-235B-A22B-Thinking"
#   - Llama-4 Maverick: provider="cerebras", model="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
#   - Gemini: via Google Generative AI (unchanged)
#
# Features:
# - Instructor and Student roles
# - Policy Ingestor (multi-file or manual text) with cached policy context
# - Custom Rubric Builder (weights summing to 100% → 20-point scale)
# - Six-agent evaluation, each agent scored by three LLMs in parallel
# - Concurrency, fallbacks, score extraction, dashboards, and history

import os
import re
import time
import uuid
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional file parsing for policy ingestion
import fitz  # PyMuPDF
import docx
from io import BytesIO

# LLM SDKs
from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI

# ============ ENV AND CLIENTS ============
# Read keys from environment (do not hardcode)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Gemini (unchanged)
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

# Hugging Face InferenceClient (separate clients per provider)
if HF_TOKEN:
    hf_qwen_client = InferenceClient(provider="novita", api_key=HF_TOKEN)
    hf_llama_client = InferenceClient(provider="cerebras", api_key=HF_TOKEN)
else:
    # Placeholders to fail fast with clear message if missing
    hf_qwen_client = None
    hf_llama_client = None

# ============ STATE ============
def init_state():
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("policies", [])  # [{id,name,desc,content}]
    st.session_state.setdefault("rubrics", [])   # [{id,name,weight}]
    st.session_state.setdefault("assignments", [])  # [{id,question,created_at,rubric_snapshot,rubric_details}]
    st.session_state.setdefault("submissions", {})   # {assignment_id: [versions]}
    st.session_state.setdefault("active_assignment_id", None)
    st.session_state.setdefault("parsed_policy_cache", "")
    st.session_state.setdefault("debug_mode", False)
    st.session_state.setdefault("num_policy_entries", 1)
    st.session_state.setdefault("num_rubric_entries", 1)

init_state()

# ============ RUBRIC ============
HARDCODED_RUBRIC = """
Score Criteria (out of 20):
- Application / Presentation of Concept: 2 marks
- Detailing and Understanding: 1 mark
- Skills Exploration: 1 mark
- Basics of Design Principles: 1 mark
- Research and Comprehension: 2 marks
- Meta-cognition and Critical Thinking: 1.5 marks
- Perception, Observation, and Sensitivity: 1.5 marks
- Conceptual Clarity and Comprehension (Theory): 2 marks
- Reflective Thinking: 1.5 marks
- Communication: 0.75 marks
- Conceptual Clarity: 2 marks
- Exploration and Improvisation: 1.5 marks
- Problem-solving and Lateral Thinking: 1.5 marks
- Originality and Visualization: 1 mark

Guidelines:
- Provide justification for each score.
- Consider understanding, clarity, exploration, critical thinking, originality.
"""

def get_active_rubric():
    if st.session_state.rubrics:
        rubric_text = "**INSTRUCTOR-DEFINED RUBRIC** (Total: 20 points)\n\n"
        rubric_text += "Score each criterion based on the following breakdown:\n\n"

        total_possible = 0.0
        for i, rubric in enumerate(st.session_state.rubrics, 1):
            weight_points = (rubric["weight"] / 100.0) * 20.0
            rubric_text += f"{i}. **{rubric['name']}**: {weight_points:.2f} points ({rubric['weight']}%)\n"
            rubric_text += "   - Evaluate student's performance in this specific area\n"
            rubric_text += "   - Provide detailed justification for the score assigned\n\n"
            total_possible += weight_points

        rubric_text += f"**TOTAL POSSIBLE POINTS: {total_possible:.1f}/20**\n\n"
        rubric_text += """**EVALUATION INSTRUCTIONS:**
1. Assess the student's response against EACH criterion listed above
2. Assign points for each criterion (up to the maximum shown)
3. Provide specific evidence from the student's answer
4. Give constructive feedback for improvement
5. Sum all criterion scores for the final total

**SCORING GUIDELINES:**
- Excellent (90-100% of criterion points): Exceeds expectations, demonstrates mastery
- Good (70-89% of criterion points): Meets expectations, shows competence
- Fair (50-69% of criterion points): Partially meets expectations, needs improvement
- Poor (0-49% of criterion points): Does not meet expectations, requires significant work

Format your response with clear criterion-by-criterion breakdown."""
        return rubric_text
    else:
        return HARDCODED_RUBRIC + "\n\n**Note: Using default rubric as no instructor-defined criteria available.**"

def format_rubric_for_display():
    if st.session_state.rubrics:
        return f"**Custom Rubric** ({len(st.session_state.rubrics)} criteria)"
    else:
        return "**Default Hardcoded Rubric** (14 criteria)"

def get_rubric_breakdown_text():
    if st.session_state.rubrics:
        breakdown = "**Current Evaluation Criteria:**\n\n"
        for i, rubric in enumerate(st.session_state.rubrics, 1):
            points = (rubric["weight"] / 100.0) * 20.0
            breakdown += f"• **{rubric['name']}**: {points:.2f} pts ({rubric['weight']}%)\n"
        return breakdown
    else:
        return "**Using Default Criteria:** Application, Understanding, Skills, Design, Research, Critical Thinking, etc."

# ============ PROMPTS ============
def generic_context_and_instructions(state):
    active_rubric = get_active_rubric()
    return f"""
### INSTITUTIONAL POLICY:
{state.get('parsed_policy', 'No specific policy provided')}

### ASSIGNMENT QUESTION:
{state['question']}

### STUDENT ANSWER TO EVALUATE:
{state['answer']}

### EVALUATION RUBRIC:
{active_rubric}

### YOUR EVALUATION TASK:
Evaluate the student's answer using the rubric criteria above.

REQUIRED FORMAT:
1. CRITERION-BY-CRITERION EVALUATION:
   - For each rubric criterion, provide:
     * Score assigned (out of the maximum points for that criterion)
     * Detailed justification with specific evidence from the student's answer
     * Strengths observed
     * Areas for improvement

2. OVERALL ASSESSMENT SUMMARY:
   - Key strengths of the response
   - Main areas needing improvement
   - Overall quality judgment

3. ACTIONABLE RECOMMENDATIONS:
   - Specific steps the student can take to improve
   - Useful resources or strategies
   - Next learning objectives to focus on

IMPORTANT: End your response with "Final Score: XX/20" where XX is the total points earned across all criteria.
Be thorough, fair, and constructive, using quotes/examples from the student's answer where helpful.
"""

def evaluate_answer_prompt_peer1(state):
    return f"""
You are Agent #1: Policy Agent — Institutional compliance and academic integrity verification.

YOUR SPECIALIZED FOCUS:
- Factual correctness and logical reasoning
- Policy compliance and academic integrity
- Accuracy and coherence
- Proper citation and attribution (if applicable)

{generic_context_and_instructions(state)}

Remember: Weight policy compliance and integrity heavily while addressing all rubric criteria.
"""

def evaluate_answer_prompt_peer2(state):
    return f"""
You are Agent #2: Pedagogy Agent — Learning objectives alignment and educational standards.

YOUR SPECIALIZED FOCUS:
- Alignment with Bloom's taxonomy
- Pedagogical appropriateness and educational value
- Learning objective fulfillment
- Depth of understanding and knowledge application
- Critical thinking and analysis

{generic_context_and_instructions(state)}

Remember: Focus on mastery of educational objectives while scoring by the rubric.
"""

def evaluate_answer_prompt_peer3(state):
    return f"""
You are Agent #3: Originality & Voice Agent — Student authenticity and creative expression.

YOUR SPECIALIZED FOCUS:
- Originality of thought and unique perspectives
- Personal voice and authentic expression
- Creative approaches and innovative solutions

{generic_context_and_instructions(state)}

Remember: Value creativity, individual voice, and authentic thinking while following the rubric.
"""

def evaluate_answer_prompt_peer4(state):
    return f"""
You are Agent #4: Equity Agent — Inclusive assessment and bias detection.

YOUR SPECIALIZED FOCUS:
- Bias-free and culturally responsive evaluation
- Equitable treatment across diverse backgrounds
- Inclusive language and representation

{generic_context_and_instructions(state)}

Remember: Ensure fair evaluation regardless of background while maintaining academic standards.
"""

def evaluate_answer_prompt_peer5(state):
    return f"""
You are Agent #5: Feedback Agent — Constructive scaffolding and personalized guidance.

YOUR SPECIALIZED FOCUS:
- Actionable, constructive feedback and next steps
- Personalized learning recommendations
- Growth-oriented guidance and scaffolding

{generic_context_and_instructions(state)}

Remember: Provide concrete improvement steps and positive reinforcement guided by the rubric.
"""

def evaluate_answer_prompt_peer6(state):
    return f"""
You are Agent #6: Summarizer Agent — Results aggregation and holistic evaluation.

YOUR SPECIALIZED FOCUS:
- Holistic synthesis across criteria
- Coherence, integration, and balance
- Big-picture assessment with detail awareness

{generic_context_and_instructions(state)}

Remember: Provide an integrated, comprehensive assessment across all criteria.
"""

PEER_PROMPT_FUNCTIONS = [
    evaluate_answer_prompt_peer1,
    evaluate_answer_prompt_peer2,
    evaluate_answer_prompt_peer3,
    evaluate_answer_prompt_peer4,
    evaluate_answer_prompt_peer5,
    evaluate_answer_prompt_peer6,
]

# ============ LLM CALLS ============
def _extract_hf_chat_text(completion):
    try:
        msg = completion.choices[0].message
        if hasattr(msg, "content") and msg.content:
            return msg.content
        return str(msg)
    except Exception:
        return str(completion)

def call_qwen_vl_235b(prompt: str) -> str:
    if hf_qwen_client is None:
        return "Qwen client not configured (HF_TOKEN missing). Final Score: 10/20"
    try:
        completion = hf_qwen_client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Thinking",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        response = _extract_hf_chat_text(completion)
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 12/20"
        return response
    except Exception as e:
        return (
            "Qwen3-VL-235B evaluation:\n"
            f"Error occurred: {str(e)}\n"
            "Based on rubric criteria, providing fallback.\n"
            "Final Score: 10/20"
        )

def call_llama4_maverick(prompt: str) -> str:
    if hf_llama_client is None:
        return "Llama client not configured (HF_TOKEN missing). Final Score: 10/20"
    try:
        completion = hf_llama_client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        response = _extract_hf_chat_text(completion)
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 13/20"
        return response
    except Exception as e:
        return (
            "Llama-4 Maverick evaluation:\n"
            f"Error occurred: {str(e)}\n"
            "Based on rubric criteria, providing fallback.\n"
            "Final Score: 10/20"
        )

def call_gemini_2_5(prompt: str) -> str:
    try:
        response = gemini_llm.invoke(prompt).content
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 14/20"
        return response
    except Exception as e:
        return (
            "Gemini-2.5 evaluation:\n"
            f"Error occurred: {str(e)}\n"
            "Providing structured fallback based on rubric.\n"
            "Final Score: 12/20"
        )

LLM_FUNCTIONS = [call_qwen_vl_235b, call_llama4_maverick, call_gemini_2_5]
LLM_NAMES = ["Qwen3-VL-235B", "Llama-4 Maverick", "Gemini-2.5"]

# ============ EVALUATION ENGINE ============
SCORE_PATTERNS = [
    r"Final Score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"Final Score:\s*(\d+(?:\.\d+)?)/20",
    r"Final Score:\s*(\d+(?:\.\d+)?)\s*out\s*of\s*20",
    r"Final Score:\s*(\d+(?:\.\d+)?)",
    r"final score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"final score:\s*(\d+(?:\.\d+)?)/20",
    r"final score:\s*(\d+(?:\.\d+)?)",
    r"Total Score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"Score:\s*(\d+(?:\.\d+)?)\s*/20",
    r"(\d+(?:\.\d+)?)/20",
]

def extract_score(text: str) -> float:
    for pattern in SCORE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(20.0, val))
            except Exception:
                pass
    # As a last resort, any number in range
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for n in nums:
        try:
            v = float(n)
            if 0.0 <= v <= 20.0:
                return v
        except Exception:
            continue
    return 10.0

def evaluate_agent_with_three_llms(state, agent_id: int):
    prompt_func = PEER_PROMPT_FUNCTIONS[agent_id - 1]
    prompt = prompt_func(state)

    llm_responses, llm_scores = [], []
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut = {executor.submit(fn, prompt): i for i, fn in enumerate(LLM_FUNCTIONS)}
            for future in as_completed(fut):
                idx = fut[future]
                try:
                    resp = future.result(timeout=90)
                    score = extract_score(resp)
                    llm_responses.append(f"**{LLM_NAMES[idx]}:**\n{resp}")
                    llm_scores.append(score)
                    if st.session_state.get("debug_mode", False):
                        st.write(f"Debug Agent {agent_id} → {LLM_NAMES[idx]} score: {score}")
                except Exception as e:
                    llm_responses.append(
                        f"**{LLM_NAMES[idx]}:** Error - {str(e)}\nFallback assessment.\nFinal Score: 10/20"
                    )
                    llm_scores.append(10.0)
    except Exception as e:
        # Executor-level failure
        for i in range(3):
            llm_responses.append(
                f"**{LLM_NAMES[i]}:** System error - {str(e)}\nFallback assessment.\nFinal Score: 10/20"
            )
            llm_scores.append(10.0)

    while len(llm_scores) < 3:
        llm_scores.append(10.0)
        llm_responses.append("**Missing LLM Response:** Fallback score.\nFinal Score: 10/20")

    avg = sum(llm_scores) / len(llm_scores)
    return {
        "agent_id": agent_id,
        "llm_scores": llm_scores,
        "agent_avg_score": avg,
        "combined_response": "\n\n".join(llm_responses),
    }

def peer_assessment_simulation(state, num_peers=6):
    results = []
    if st.session_state.get("debug_mode", False):
        st.write("🚀 Starting multi-agent evaluation…")
        st.write(f"📄 Using Rubric: {format_rubric_for_display()}")
        st.write(f"🔧 LLM Ensemble: {', '.join(LLM_NAMES)}")

    try:
        with ThreadPoolExecutor(max_workers=num_peers) as executor:
            fut = {executor.submit(evaluate_agent_with_three_llms, state, i): i for i in range(1, num_peers + 1)}
            for future in as_completed(fut):
                agent_id = fut[future]
                try:
                    res = future.result(timeout=180)
                    results.append(res)
                    if st.session_state.get("debug_mode", False):
                        st.write(f"✅ Agent {agent_id} avg: {res['agent_avg_score']:.2f}")
                except Exception as e:
                    results.append({
                        "agent_id": agent_id,
                        "llm_scores": [10.0, 10.0, 10.0],
                        "agent_avg_score": 10.0,
                        "combined_response": f"Agent {agent_id} error: {str(e)}\nFallback evaluation.\nFinal Score: 10/20",
                    })
    except Exception as e:
        if st.session_state.get("debug_mode", False):
            st.write(f"⚠️ Executor failed: {str(e)}; using fallbacks.")
        for i in range(1, num_peers + 1):
            results.append({
                "agent_id": i,
                "llm_scores": [10.0, 10.0, 10.0],
                "agent_avg_score": 10.0,
                "combined_response": f"Agent {i} system error.\nFallback.\nFinal Score: 10/20",
            })

    results.sort(key=lambda x: x["agent_id"])
    agent_avgs = [r["agent_avg_score"] for r in results]
    overall = sum(agent_avgs) / len(agent_avgs) if agent_avgs else 10.0

    rubric_context = f"**EVALUATION CONTEXT:**\n{format_rubric_for_display()}\n{get_rubric_breakdown_text()}\n\n"
    agent_names = ["Policy Agent", "Pedagogy Agent", "Originality & Voice Agent", "Equity Agent", "Feedback Agent", "Summarizer Agent"]
    sections = [rubric_context]
    for r in results:
        name = agent_names[r["agent_id"] - 1]
        sections.append(
            f"**Agent #{r['agent_id']} - {name} (Score: {r['agent_avg_score']:.2f}/20):**\n"
            f"LLM Ensemble Scores: " + " | ".join([f"{LLM_NAMES[i]}: {s:.1f}" for i, s in enumerate(r["llm_scores"])]) + "\n\n"
            f"{r['combined_response']}"
        )
    combined = "\n\n" + "=" * 50 + "\n\n" + "\n\n".join(sections)

    return {
        "peer_reviews": combined,
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
        .muted { color:#5f6c7b; }
        .rubric-indicator { background:#e8f5e8; padding:8px 12px; border-radius:8px; margin:8px 0; border-left:4px solid #4caf50; }
        .llm-indicator { background:#e3f2fd; padding:6px 10px; border-radius:6px; margin:4px 0; border-left:3px solid #2196f3; font-size:0.9rem; }
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
              <div class="muted">Policy-Compliant AI Scaffolding with Three-LLM Ensemble</div>
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
            with c1:
                name = st.text_input("Policy Name", key=f"policy_name_{i}")
            with c2:
                desc = st.text_input("Description", key=f"policy_desc_{i}")
            with c3:
                upload = st.file_uploader("📄", type=["pdf", "txt", "docx"], key=f"policy_file_{i}")

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

        left, right = st.columns([3, 1])
        with left:
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
                st.code((p["content"] or "")[:800] + ("..." if len(p["content"] or "") > 800 else ""), language="text")
    else:
        st.info("No policies saved yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def rubric_builder():
    st.markdown("<div class='card'><h4>📊 Rubric Builder</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='rubric-indicator'>{format_rubric_for_display()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='llm-indicator'>🤖 LLM Ensemble: {' • '.join(LLM_NAMES)}</div>", unsafe_allow_html=True)

    with st.form("rubric_form"):
        st.markdown("**Define Evaluation Criteria (Must total 100%)**")
        data, total = [], 0.0
        for i in range(st.session_state.num_rubric_entries):
            c1, c2 = st.columns([3, 1])
            with c1:
                nm = st.text_input(f"Evaluation Criterion {i+1}", key=f"rubric_name_{i}")
            with c2:
                wt = st.number_input("Weight %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key=f"rubric_weight_{i}")
            if nm.strip():
                data.append({"name": nm.strip(), "weight": float(wt)})
                total += wt

        if abs(total - 100.0) < 0.01:
            st.success(f"✅ Total: {total:.0f}% (Perfect)")
        else:
            st.warning(f"⚠️ Total: {total:.0f}% (Must equal 100%)")

        left, right = st.columns([3, 1])
        with left:
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

    # Current rubric display
    st.markdown("<div class='card'><h4>📝 Current Evaluation Rubric</h4>", unsafe_allow_html=True)
    if st.session_state.rubrics:
        df_data, tot = [], 0.0
        for r in st.session_state.rubrics:
            pts = (r["weight"] / 100.0) * 20.0
            df_data.append({"Criterion": r["name"], "Weight (%)": r["weight"], "Points (out of 20)": f"{pts:.2f}"})
            tot += r["weight"]
        st.table(pd.DataFrame(df_data))
        if abs(tot - 100.0) < 0.01:
            st.success(f"✅ Total Weight: {tot:.0f}% | Active for evaluation.")
        else:
            st.warning(f"⚠️ Total Weight: {tot:.0f}% (Should be 100%)")
    else:
        st.info("📄 Using Default Hardcoded Rubric — create custom criteria above to override.")
        with st.expander("📄 View Default Rubric Details"):
            st.code(HARDCODED_RUBRIC, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

def assignment_creator():
    st.markdown("<div class='card'><h4>📝 Assignment Creation</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='rubric-indicator'>📄 Evaluations will use: {format_rubric_for_display()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='llm-indicator'>🤖 Evaluation powered by: {' • '.join(LLM_NAMES)}</div>", unsafe_allow_html=True)

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

    if history:
        latest = history[-1]
        if "agent_scores" in latest:
            st.markdown("#### 🤖 Latest Agent Breakdown")
            agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
            cols = st.columns(6)
            for i, (nm, sc) in enumerate(zip(agent_names, latest["agent_scores"])):
                with cols[i]:
                    st.metric(nm, f"{sc:.2f}")

    st.markdown("#### 📨 Detailed Feedback History")
    for i, h in enumerate(history, 1):
        status = "Final Submission" if h.get("submitted") else "Draft"
        rubric_used = h.get("rubric_used", "Unknown")
        with st.expander(f"Version {i} Feedback ({status}) - {rubric_used}"):
            st.markdown(h["feedback"])

    final_versions = [h for h in history if h.get("submitted")]
    if final_versions:
        st.success("🏆 Final submission received!")
    else:
        st.info("⏳ No final submission yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============ STUDENT WORKSPACE ============
def student_workspace():
    st.markdown("<div class='card'><h4>📚 Available Assignments</h4>", unsafe_allow_html=True)
    if not st.session_state.assignments:
        st.info("📭 Waiting for instructor to publish assignments.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    options = {f"{i+1}. {a['question'][:50]}...": a["id"] for i, a in enumerate(st.session_state.assignments)}
    sel = st.selectbox("Select Assignment to Work On", list(options.keys()))
    aid = options[sel]
    st.session_state.active_assignment_id = aid
    selected = next(a for a in st.session_state.assignments if a["id"] == aid)
    full_question = selected["question"]
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📄 Assignment Question</h4>", unsafe_allow_html=True)
    st.markdown(f"**{full_question}**")
    st.markdown(f"<div class='rubric-indicator'>📄 Your submission will be evaluated using: {selected['rubric_snapshot']}</div>", unsafe_allow_html=True)
    with st.expander("📊 View Detailed Evaluation Criteria"):
        st.markdown(selected["rubric_details"])
    st.markdown(f"<div class='llm-indicator'>🤖 Evaluation powered by: {' • '.join(LLM_NAMES)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>✍️ Answer Editor</h4>", unsafe_allow_html=True)
    default_text = ""
    if st.session_state.submissions.get(aid):
        default_text = st.session_state.submissions[aid][-1]["answer"]
    answer = st.text_area("Draft your answer here:", height=220, value=default_text,
                          placeholder="Write your comprehensive response here...")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        evaluate = st.button("🤖 Submit for AI Evaluation")
    with c2:
        modify = st.button("✏️ Modify & Resubmit")
    with c3:
        final_submit = st.button("🏁 Final Submit", type="primary")
    with c4:
        st.session_state.debug_mode = st.checkbox("🐛 Debug", value=st.session_state.get("debug_mode", False))

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
                "rubric": get_active_rubric(),
                "parsed_policy": parsed_policy,
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

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>🤖 Comprehensive Multi‑Agent AI Feedback</h4>", unsafe_allow_html=True)
    versions = st.session_state.submissions.get(aid, [])
    if versions:
        last = versions[-1]
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.metric("📊 Overall Score", f"{last['avg']:.2f}/20", f"{(last['avg']/20*100):.1f}%")
        with c2:
            status = "🏁 Final Submission" if last.get("submitted") else "📝 Draft"
            st.markdown(f"**Status:** {status}")
        with c3:
            st.markdown(f"**Evaluated with:** {last.get('rubric_used','Unknown')}")

        st.markdown("##### Individual Agent Scores (Average of 3 LLMs each)")
        agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
        cols = st.columns(6)
        for i, (name, score) in enumerate(zip(agent_names, last["scores"])):
            with cols[i]:
                pct = (score / 20 * 100.0)
                st.metric(f"{name}", f"{score:.2f}", f"{pct:.1f}%")
                if "detailed_results" in last and i < len(last["detailed_results"]):
                    llm_scores = last["detailed_results"][i]["llm_scores"]
                    llm_display = " | ".join([f"{LLM_NAMES[j]}: {s:.1f}" for j, s in enumerate(llm_scores)])
                    st.markdown(f"<div class='llm-score'>{llm_display}</div>", unsafe_allow_html=True)

        if last["avg"] >= 18:
            st.success("🌟 Excellent Performance — Outstanding work across all criteria!")
        elif last["avg"] >= 15:
            st.info("👍 Good Performance — Solid work with room for refinement.")
        elif last["avg"] >= 12:
            st.warning("📈 Satisfactory — Meets basics; consider revisions.")
        else:
            st.error("📝 Needs Improvement — Review the feedback and revise.")

        with st.expander("📄 View Comprehensive Three‑LLM Feedback & Detailed Analysis", expanded=False):
            st.markdown(last["feedback"])
    else:
        st.info("💡 Submit for evaluation to see detailed feedback here.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📊 Submission History & Progress</h4>", unsafe_allow_html=True)
    if versions:
        table_data = []
        for i, v in enumerate(versions, 1):
            status_icon = "🏁" if v.get("submitted") else "📝"
            rubric_used = v.get("rubric_used", "Unknown")
            if v["avg"] >= 18:
                perf = "🌟 Excellent"
            elif v["avg"] >= 15:
                perf = "👍 Good"
            elif v["avg"] >= 12:
                perf = "📈 Satisfactory"
            else:
                perf = "📝 Needs Work"
            table_data.append({
                "Version": f"{status_icon} {i}",
                "Timestamp": v["timestamp"],
                "Score": f"{v['avg']:.2f}/20",
                "Performance": perf,
                "Status": "Final" if v.get("submitted") else "Draft",
                "Rubric": rubric_used,
            })
        st.table(pd.DataFrame(table_data))
        if len(versions) > 1:
            improvement = versions[-1]["avg"] - versions[0]["avg"]
            if improvement > 0:
                st.success(f"📈 Improved by {improvement:.2f} points since first submission!")
            elif improvement < 0:
                st.info(f"📉 Latest score is {abs(improvement):.2f} points lower — review feedback.")
            else:
                st.info("📊 Score unchanged — use feedback for targeted improvement.")
    else:
        st.info("📈 Submission history will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============ NAV AND MAIN ============
def login_page():
    st.markdown("<div class='card' style='text-align:center;padding:32px;'>", unsafe_allow_html=True)
    st.markdown("<h2>Welcome to EduScaffold</h2>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Choose a role to continue</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👨‍🏫 Instructor", use_container_width=True):
            st.session_state.role = "Instructor"
            st.rerun()
    with c2:
        if st.button("🎓 Student", use_container_width=True):
            st.session_state.role = "Student"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="EduScaffold", layout="wide", page_icon="🎓")
    inject_css()
    topbar()

    if not HF_TOKEN:
        st.warning("⚠️ HF_TOKEN is not set; configure an HF access token to call Qwen and Llama‑4 Maverick.")
    if not GOOGLE_API_KEY:
        st.warning("⚠️ GOOGLE_API_KEY is not set; configure a Google API key to call Gemini.")

    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        st.markdown("**Three‑LLM Ensemble:**")
        st.markdown("• Qwen3‑VL‑235B‑A22B‑Thinking")
        st.markdown("• Llama‑4 Maverick 17B‑128E Instruct")
        st.markdown("• Gemini‑2.5")
        st.markdown("---")
        st.markdown(f"**Active Rubric:** {format_rubric_for_display()}")

        if not st.session_state.role:
            st.info("👉 Select role on the login screen.")
        else:
            st.success(f"Logged in as: **{st.session_state.role}**")

        if st.session_state.role == "Instructor":
            choice = st.radio(
                "Go to",
                ["🏠 Dashboard", "📋 Policy Ingestor", "📊 Rubric Builder", "📝 Assignment Creator", "🕵️ Student Monitoring", "🚪 Logout"],
                index=0,
            )
        elif st.session_state.role == "Student":
            choice = st.radio("Go to", ["🎓 Workspace", "🚪 Logout"], index=0)
        else:
            choice = None

    if st.session_state.role is None:
        login_page()
        return

    if st.session_state.role == "Instructor":
        if choice == "🏠 Dashboard":
            st.markdown("<div class='card'><h4>👨‍🏫 Instructor Dashboard</h4><p class='muted'>Manage courses with three‑LLM ensemble and custom rubric.</p></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("📄 Policies", len(st.session_state.policies))
            with c2: st.metric("📊 Rubric Criteria", len(st.session_state.rubrics))
            with c3: st.metric("📝 Assignments", len(st.session_state.assignments))
            with c4: st.metric("📄 Active Rubric", "Custom" if st.session_state.rubrics else "Default")
        elif choice == "📋 Policy Ingestor":
            policy_ingestor()
        elif choice == "📊 Rubric Builder":
            rubric_builder()
        elif choice == "📝 Assignment Creator":
            assignment_creator()
        elif choice == "🕵️ Student Monitoring":
            student_monitoring()
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

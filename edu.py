# EduScaffold Frontend (Streamlit) - Updated with Dynamic Forms
# Modern academic UI with Instructor and Student views, shared state, and multi-agent evaluation

import os
import re
import time
import uuid
import streamlit as st
import pandas as pd

# Optional: PDF parsing
import fitz
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCf4zhRvpmANps5SVaIPOZ3QoZnHA5SNlw"
# ============ THEME / LLM SETUP ============
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCf4zhRvpmANps5SVaIPOZ3QoZnHA5SNlw")
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

# ============ MULTI-AGENT EVALUATION (from provided snippet, adapted) ============
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

def generic_context_and_instructions(state):
    return f"""
### Institutional Policy:
{state.get('parsed_policy', 'No policy provided')}

### Question:
{state['question']}

### Student Answer:
{state['answer']}

### Rubric:
{state['rubric']}

Instructions:
1. Score the answer on a 20-point scale.
2. Justify the score thoroughly using rubric criteria and relevant policy rules.
3. Provide detailed feedback on:
   - Accuracy
   - Clarity
   - Policy compliance
   - Strengths and weaknesses
   - Recommendations for improvement

Be strict and insightful. Provide the final score clearly like: Final Score: XX/20
"""

def evaluate_answer_prompt_peer1(state):
    return f"""
You are Peer Reviewer #1: An evaluator focused on conceptual accuracy and depth.
Your job:
- Check for factual correctness and logic
- Judge conceptual depth and understanding
{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer2(state):
    return f"""
You are Peer Reviewer #2: An evaluator focused on grammar, clarity, communication, and organization.
Your job:
- Judge the grammatical accuracy of the answer.
- Assess sentence structure, punctuation, and proper academic expression.
- Check coherence, logical flow, and clarity of ideas.
- Highlight areas where grammar or clarity impacts understanding.
You may point out specific grammar mistakes or suggest improved phrasing.
{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer3(state):
    return f"""
You are Peer Reviewer #3: A rubric and institutional policy compliance evaluator.
Your job:
- Enforce policy and rubric strictly
- Penalize deviation from expected formats or standards
{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer4(state):
    return f"""
You are Peer Reviewer #4: A critical thinking and reasoning specialist.
Your job:
- Evaluate originality of thought
- Look for logical analysis and reasoning quality
{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer5(state):
    return f"""
You are Peer Reviewer #5: A creativity, visualization, and problem-solving analyst.
Your job:
- Reward lateral thinking and creative approach
- Look for use of analogies or unique ideas
{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer6(state):
    return f"""
You are Peer Reviewer #6: A student development and growth mindset advocate.
Your job:
- Provide constructive, motivating feedback
- Evaluate learning mindset and progress orientation
{generic_context_and_instructions(state)}
"""

PEER_PROMPT_FUNCTIONS = [
    evaluate_answer_prompt_peer1,
    evaluate_answer_prompt_peer2,
    evaluate_answer_prompt_peer3,
    evaluate_answer_prompt_peer4,
    evaluate_answer_prompt_peer5,
    evaluate_answer_prompt_peer6
]

def evaluate_single_peer(state, peer_id=1):
    prompt_func = PEER_PROMPT_FUNCTIONS[peer_id - 1]
    prompt = prompt_func(state)
    try:
        response = gemini_llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"Peer Reviewer #{peer_id} failed to respond. Final Score: 0/20\nError: {e}"

def peer_assessment_simulation(state, num_peers=6):
    peer_evaluations = []
    scores = []
    for i in range(1, num_peers + 1):
        evaluation_text = evaluate_single_peer(state, peer_id=i)
        peer_evaluations.append((i, evaluation_text))
        match = re.search(r"(?i)\b(?:final\s*)?score\s*[:\-]?\s*(\d+(\.\d+)?)(?:\s*/20|\s*\/20)?", evaluation_text, re.IGNORECASE)
        if match:
            scores.append(float(match.group(1)))
        else:
            scores.append(0)
    avg_score = sum(scores) / len(scores) if scores else 0
    combined_feedback = "\n\n".join(
        [f"**Peer Reviewer #{peer_id}:**\n{eval_text}" for peer_id, eval_text in peer_evaluations]
    )
    return {
        "peer_reviews": combined_feedback,
        "average_score": avg_score,
        "scores": scores,
    }

# ============ SHARED STATE & MODELS ============
def init_state():
    st.session_state.setdefault("role", None)  # "Instructor" or "Student"
    st.session_state.setdefault("policies", [])  # list of {id,name,desc,content}
    st.session_state.setdefault("rubrics", [])   # list of {id,name,weight}
    st.session_state.setdefault("assignments", [])  # list of {id,question,created_at}
    st.session_state.setdefault("submissions", {})  # map assignment_id -> list of versions [{answer, feedback, scores, avg, timestamp, submitted}]
    st.session_state.setdefault("active_assignment_id", None)
    st.session_state.setdefault("parsed_policy_cache", "")  # concat of policy content for evaluation context
    
    # Dynamic form states
    st.session_state.setdefault("num_policy_entries", 1)
    st.session_state.setdefault("num_rubric_entries", 1)

init_state()

# ============ STYLING ============
def inject_css():
    st.markdown(
        """
        <style>
        /* Background gradient: soft yellow, green, blue */
        .stApp {
            background: linear-gradient(135deg, #FFFBEA 0%, #EAF8F0 40%, #E9F3FF 100%);
        }
        /* Card aesthetic */
        .card {
            background: #ffffffAA;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.05);
            margin-bottom: 14px;
        }
        .card h4 { margin-top: 0.2rem; }
        .muted { color: #5f6c7b; }
        .pill {
            display:inline-block; padding:6px 10px; border-radius:999px;
            background:#EEF6FF; color:#1f4e79; font-size: 0.78rem; font-weight:600;
        }
        .primary-btn {
            background: linear-gradient(135deg, #6CC7B0, #6FB8FF);
            color: white; border:none; border-radius: 10px; padding: 10px 14px; font-weight: 600;
        }
        .accent-btn {
            background: linear-gradient(135deg, #FFEB99, #B7F0C0);
            color: #1b2a3a; border:none; border-radius: 10px; padding: 10px 14px; font-weight: 600;
        }
        .danger-btn {
            background: linear-gradient(135deg, #FF8A8A, #FFC0C0);
            color: #1b2a3a; border:none; border-radius: 10px; padding: 10px 14px; font-weight: 600;
        }
        .header {
            display:flex; align-items:center; justify-content:space-between;
            padding: 8px 6px; margin-bottom: 8px;
        }
        .brand {
            display:flex; align-items:center; gap:10px;
        }
        .logo {
            width:36px; height:36px; border-radius:8px;
            background: linear-gradient(135deg, #FFF5B1, #BDE8C6, #BFE1FF);
            display:flex; align-items:center; justify-content:center; font-weight:800; color:#1f4e79;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .userchip {
            display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:999px;
            background:#ffffff66; backdrop-filter: blur(6px);
            box-shadow: 0 1px 10px rgba(0,0,0,0.04);
        }
        .title-xl { font-size: 1.5rem; font-weight:800; color:#1f4e79; }
        .subtle { font-size: 0.92rem; color:#3b4a5a; }
        /* Sidebar tweaks */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFBEA 0%, #EAF8F0 60%, #E9F3FF 100%);
        }
        .percentage-warning {
            color: #ff6b6b;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .percentage-success {
            color: #51cf66;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .form-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .add-btn {
            background: linear-gradient(135deg, #FFEB99, #B7F0C0);
            border: none;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 20px;
            font-weight: bold;
            color: #1f4e79;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def topbar():
    st.markdown(
        """
        <div class="header">
          <div class="brand">
            <div class="logo">ES</div>
            <div>
              <div class="title-xl">EduScaffold</div>
              <div class="subtle">Policy-Compliant AI Scaffolding</div>
            </div>
          </div>
          <div class="userchip">
            <span>🔔</span>
            <span>👤</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============ LOGIN PAGE ============
def login_page():
    st.markdown("<div class='card' style='text-align:center;padding:40px;'>", unsafe_allow_html=True)
    st.markdown("<div class='logo' style='margin:0 auto 10px;'>ES</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:0;'>Welcome to EduScaffold</h2>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Choose a role to continue</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        colA, colB = st.columns(2)
        with colA:
            if st.button("👨‍🏫 Instructor", use_container_width=True):
                st.session_state.role = "Instructor"
                st.rerun()
        with colB:
            if st.button("👨‍🎓 Student", use_container_width=True):
                st.session_state.role = "Student"
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============ INSTRUCTOR MODULES ============
def policy_ingestor():
    st.markdown("<div class='card'><h4>📋 Policy Ingestor</h4>", unsafe_allow_html=True)
    
    # Dynamic policy entries
    with st.form("policy_form"):
        policies_data = []
        
        for i in range(st.session_state.num_policy_entries):
            st.markdown(f"**Policy Entry {i+1}**")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                name = st.text_input(f"Policy Name", key=f"policy_name_{i}")
                
            with col2:
                desc = st.text_input(f"Description", key=f"policy_desc_{i}")
                
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input fields
                uploaded_file = st.file_uploader(f"📄", type=["pdf", "txt", "docx"], key=f"policy_file_{i}")
            
            # Text content area for manual entry or extracted text
            content = ""
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    try:
                        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        content = "\n".join([page.get_text() for page in doc])
                        st.success(f"✅ PDF content extracted for Policy {i+1}")
                    except Exception as e:
                        st.error(f"❌ Error reading PDF: {str(e)}")
                else:
                    content = str(uploaded_file.read(), "utf-8")
                    st.success(f"✅ File content loaded for Policy {i+1}")
            
            manual_content = st.text_area(f"Policy Content (Manual Entry or Extracted)", 
                                        value=content, height=100, key=f"policy_content_{i}")
            
            policies_data.append({
                "name": name,
                "desc": desc, 
                "content": manual_content
            })
            
            st.markdown("---")
        
        # Form buttons
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.form_submit_button("💾 Save All Policies", type="primary"):
                saved_count = 0
                for policy_data in policies_data:
                    if policy_data["name"].strip() and (policy_data["desc"].strip() or policy_data["content"].strip()):
                        st.session_state.policies.append({
                            "id": str(uuid.uuid4()),
                            "name": policy_data["name"].strip(),
                            "desc": policy_data["desc"].strip(),
                            "content": policy_data["content"].strip()
                        })
                        saved_count += 1
                
                if saved_count > 0:
                    # Refresh parsed policy cache
                    st.session_state.parsed_policy_cache = "\n\n".join([p["content"] for p in st.session_state.policies])
                    st.success(f"✅ {saved_count} policies saved successfully!")
                else:
                    st.warning("⚠️ Please provide at least name and description/content for each policy.")
    
    # Add more button (outside form to avoid form reset)
    if st.button("➕ Add More Policy Entry"):
        st.session_state.num_policy_entries += 1
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Display saved policies
    st.markdown("<div class='card'><h4>📚 Saved Policies</h4>", unsafe_allow_html=True)
    if st.session_state.policies:
        for i, p in enumerate(st.session_state.policies):
            with st.expander(f"📋 {p['name']}"):
                st.write(f"**Description:** {p['desc'] or '(No description)'}")
                st.code(p["content"][:500] + ("..." if len(p["content"]) > 500 else ""), language="text")
    else:
        st.info("No policies saved yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def rubric_builder():
    st.markdown("<div class='card'><h4>📊 Rubric Builder</h4>", unsafe_allow_html=True)
    
    with st.form("rubric_form"):
        rubrics_data = []
        total_percentage = 0
        
        for i in range(st.session_state.num_rubric_entries):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                rubric_name = st.text_input(f"Rubric Criterion", 
                                          placeholder=f"e.g., Critical Thinking",
                                          key=f"rubric_name_{i}")
            with col2:
                weight = st.number_input(f"Weight %", 
                                       min_value=0.0, max_value=100.0, 
                                       value=10.0, step=1.0,
                                       key=f"rubric_weight_{i}")
            
            if rubric_name.strip():
                rubrics_data.append({
                    "name": rubric_name.strip(),
                    "weight": float(weight)
                })
                total_percentage += weight
        
        # Percentage validation display
        if total_percentage == 100:
            st.markdown(f'<p class="percentage-success">✅ Total: {total_percentage}% (Perfect!)</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="percentage-warning">⚠️ Total: {total_percentage}% (Must equal 100%)</p>', unsafe_allow_html=True)
        
        # Form buttons
        col1, col2 = st.columns([3, 1])
        with col1:
            save_rubrics = st.form_submit_button("💾 Save Rubrics", type="primary")
            
        if save_rubrics:
            if abs(total_percentage - 100) < 0.01:  # Allow small floating point errors
                saved_count = 0
                for rubric_data in rubrics_data:
                    if rubric_data["name"]:
                        st.session_state.rubrics.append({
                            "id": str(uuid.uuid4()),
                            "name": rubric_data["name"],
                            "weight": rubric_data["weight"]
                        })
                        saved_count += 1
                
                if saved_count > 0:
                    st.success(f"✅ {saved_count} rubric criteria saved!")
                else:
                    st.warning("⚠️ Please provide names for rubric criteria.")
            else:
                st.error("❌ Rubric weights must total exactly 100%!")
    
    # Add more button (outside form)
    if st.button("➕ Add More Rubric Entry"):
        st.session_state.num_rubric_entries += 1
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Display saved rubrics
    st.markdown("<div class='card'><h4>📈 Current Rubrics</h4>", unsafe_allow_html=True)
    if st.session_state.rubrics:
        df_data = []
        total_weight = 0
        for r in st.session_state.rubrics:
            df_data.append({"Criterion": r["name"], "Weight (%)": r["weight"]})
            total_weight += r["weight"]
        
        df = pd.DataFrame(df_data)
        st.table(df)
        
        if abs(total_weight - 100) < 0.01:
            st.success(f"✅ Total Weight: {total_weight}%")
        else:
            st.warning(f"⚠️ Total Weight: {total_weight}% (Should be 100%)")
    else:
        st.info("No rubric criteria saved yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def assignment_creator():
    st.markdown("<div class='card'><h4>📝 Assignment Creation</h4>", unsafe_allow_html=True)
    with st.form("assignment_form", clear_on_submit=True):
        q = st.text_area("📋 Question / Prompt", height=120, 
                        placeholder="Enter the assignment question or prompt here...")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.form_submit_button("🚀 Publish Assignment", type="primary"):
                if q.strip():
                    st.session_state.assignments.append({
                        "id": str(uuid.uuid4()),
                        "question": q.strip(),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success("✅ Assignment published to Student Workspace!")
                else:
                    st.warning("⚠️ Please enter a question.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📚 Published Assignments</h4>", unsafe_allow_html=True)
    if st.session_state.assignments:
        for i, a in enumerate(st.session_state.assignments, 1):
            st.markdown(f"**{i}.** {a['question']}")
            st.markdown(f"<small class='muted'>📅 Published: {a['created_at']}</small>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No assignments published yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def student_monitoring():
    st.markdown("<div class='card'><h4>👁️ Student Monitoring Dashboard</h4>", unsafe_allow_html=True)
    if not st.session_state.assignments:
        st.info("📭 No assignments published yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Select assignment
    options = {f"{i+1}. {a['question'][:50]}...": a["id"] for i, a in enumerate(st.session_state.assignments)}
    sel = st.selectbox("📋 Select Assignment to Monitor", list(options.keys()))
    aid = options[sel]
    history = st.session_state.submissions.get(aid, [])

    if not history:
        st.info("📝 No student submissions yet for this assignment.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Timeline
    st.markdown("#### 🕐 Submission Timeline")
    for i, h in enumerate(history, start=1):
        status_icon = "🏁" if h.get('submitted') else "📝"
        st.markdown(f"{status_icon} **Version {i}** — {h['timestamp']} — Score: {h['avg']:.1f}/20")

    # Score progression
    st.markdown("#### 📈 Score Progression")
    scores = [h["avg"] for h in history]
    if scores:
        df = pd.DataFrame({"Version": list(range(1, len(scores)+1)), "Score": scores})
        st.line_chart(df.set_index("Version"))
    else:
        st.write("No scores yet.")

    # Feedback logs
    st.markdown("#### 💭 Feedback History")
    for i, h in enumerate(history, start=1):
        status = "Final Submission" if h.get('submitted') else "Draft"
        with st.expander(f"Version {i} Feedback ({status})"):
            st.markdown(h["feedback"])

    # Final submission indicator
    final_versions = [h for h in history if h.get("submitted")]
    if final_versions:
        st.success("🎯 Final submission received!")
        if st.button("📖 Review Final Work"):
            v = final_versions[-1]
            st.markdown("##### 📝 Final Answer")
            st.write(v["answer"])
            st.markdown("##### 💭 Final Feedback")
            st.markdown(v["feedback"])
            st.metric("Final Score", f"{v['avg']:.2f}/20")
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
    
    # Get full question
    full_question = next(a["question"] for a in st.session_state.assignments if a["id"] == aid)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📋 Assignment Question</h4>", unsafe_allow_html=True)
    st.markdown(f"**{full_question}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Answer editor + feedback
    st.markdown("<div class='card'><h4>✍️ Answer Editor</h4>", unsafe_allow_html=True)
    default_text = ""
    if st.session_state.submissions.get(aid):
        default_text = st.session_state.submissions[aid][-1]["answer"]
    
    answer = st.text_area("Draft your answer here:", height=180, value=default_text)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        evaluate = st.button("🤖 Submit for AI Evaluation")
    with col2:
        modify = st.button("✏️ Modify & Resubmit")
    with col3:
        final_submit = st.button("🎯 Final Submit", type="primary")

    # Ensure policy string is available for context
    parsed_policy = st.session_state.parsed_policy_cache or "No policy provided"
    
    if evaluate or modify:
        if not answer.strip():
            st.warning("⚠️ Please enter an answer before evaluation.")
        else:
            eval_state = {
                "question": full_question,
                "answer": answer,
                "rubric": HARDCODED_RUBRIC,
                "parsed_policy": parsed_policy
            }
            with st.spinner("🤖 Running multi-agent evaluation..."):
                results = peer_assessment_simulation(eval_state, num_peers=6)

            # Prepare feedback display
            peer_texts = []
            scores = []
            for i in range(1, 7):
                match = re.search(
                    fr"\*\*Peer Reviewer #{i}:\*\*\n(.+?)(?=(\*\*Peer Reviewer #\d+:\*\*|$))",
                    results["peer_reviews"],
                    flags=re.DOTALL,
                )
                feedback = match.group(1).strip() if match else "No feedback."
                peer_texts.append(feedback)
                score_match = re.search(r"(?i)(?:final\s*)?score\s*[:\-]?\s*(\d+(\.\d+)?)/20", feedback)
                scores.append(float(score_match.group(1)) if score_match else 0.0)
            
            avg = sum(scores)/len(scores) if scores else 0.0

            # Save version
            st.session_state.submissions.setdefault(aid, []).append({
                "answer": answer,
                "scores": scores,
                "avg": avg,
                "feedback": results["peer_reviews"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "submitted": False
            })
            st.success("✅ Evaluation complete! Check feedback below.")

    # Final submit confirmation
    if final_submit:
        versions = st.session_state.submissions.get(aid, [])
        if not versions:
            st.warning("⚠️ Please evaluate at least once before final submission.")
        else:
            versions[-1]["submitted"] = True
            st.success("🎯 Final submission recorded! Your instructor can now review it.")

    st.markdown("</div>", unsafe_allow_html=True)

    # AI Feedback panel
    st.markdown("<div class='card'><h4>🤖 AI Multi-Agent Feedback</h4>", unsafe_allow_html=True)
    versions = st.session_state.submissions.get(aid, [])
    if versions:
        last = versions[-1]
        
        # Score display
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("📊 Average Score", f"{last['avg']:.1f}/20")
        with col2:
            status = "🎯 Final Submission" if last.get('submitted') else "📝 Draft"
            st.markdown(f"**Status:** {status}")
        
        # Individual scores
        st.markdown("##### Individual Agent Scores")
        score_cols = st.columns(6)
        for i, score in enumerate(last['scores']):
            with score_cols[i]:
                st.metric(f"Agent {i+1}", f"{score:.1f}")
        
        # Detailed feedback
        with st.expander("📝 View Detailed Feedback & Suggestions", expanded=False):
            st.markdown(last["feedback"])
    else:
        st.info("💭 Submit your answer for AI evaluation to see feedback here.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Version history
    st.markdown("<div class='card'><h4>📊 Version History</h4>", unsafe_allow_html=True)
    if versions:
        table_data = []
        for i, v in enumerate(versions, start=1):
            status_icon = "🎯" if v.get("submitted") else "📝"
            table_data.append({
                "Version": f"{status_icon} {i}",
                "Timestamp": v["timestamp"],
                "Score": f"{v['avg']:.1f}/20",
                "Status": "Final" if v.get("submitted") else "Draft"
            })
        st.table(pd.DataFrame(table_data))
    else:
        st.info("📈 Your submission history will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============ MAIN APP ============
def main():
    st.set_page_config(page_title="EduScaffold", layout="wide", page_icon="🎓")
    inject_css()
    topbar()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        if not st.session_state.role:
            st.info("👈 Select role on the login screen.")
        else:
            st.success(f"Logged in as: **{st.session_state.role}**")
            
        if st.session_state.role == "Instructor":
            choice = st.radio(
                "Go to",
                ["🏠 Dashboard", "📋 Policy Ingestor", "📊 Rubric Builder", 
                 "📝 Assignment Creator", "👁️ Student Monitoring", "🚪 Logout"],
                index=0
            )
        elif st.session_state.role == "Student":
            choice = st.radio(
                "Go to",
                ["🎓 Workspace", "🚪 Logout"],
                index=0
            )
        else:
            choice = None

    # Router
    if st.session_state.role is None:
        login_page()
        return

    if st.session_state.role == "Instructor":
        if choice == "🏠 Dashboard":
            st.markdown("<div class='card'><h4>👨‍🏫 Instructor Dashboard</h4><p class='muted'>Welcome! Use the sidebar to access different modules and manage your courses.</p></div>", unsafe_allow_html=True)
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Policies", len(st.session_state.policies))
            with col2:
                st.metric("📊 Rubrics", len(st.session_state.rubrics))
            with col3:
                st.metric("📝 Assignments", len(st.session_state.assignments))
                
        elif choice == "📋 Policy Ingestor":
            policy_ingestor()
        elif choice == "📊 Rubric Builder":
            rubric_builder()
        elif choice == "📝 Assignment Creator":
            assignment_creator()
        elif choice == "👁️ Student Monitoring":
            student_monitoring()
        elif choice == "🚪 Logout":
            st.session_state.role = None
            # Reset dynamic form counters
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

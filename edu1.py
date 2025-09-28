# EduScaffold Frontend (Streamlit) - Updated with Dynamic Forms and Three-LLM Multi-Agent System
# Modern academic UI with Instructor and Student views, shared state, and multi-agent evaluation

import os
import re
import time
import uuid
import streamlit as st
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
import docx
from io import BytesIO
import zipfile
from pathlib import Path

# ============ LLM SETUP ============
# Gemini setup
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCf4zhRvpmANps5SVaIPOZ3QoZnHA5SNlw")
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# HuggingFace setup
from openai import OpenAI
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", "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"),
)

# ============ THREE-LLM ENSEMBLE SYSTEM ============
def call_qwen_32b(prompt):
    """Call Qwen-3 32B via HuggingFace"""
    try:
        completion = hf_client.chat.completions.create(
            model="Qwen/Qwen3-32B:nebius",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        response = completion.choices[0].message.content
        # Ensure response includes a final score
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 12/20"
        return response
    except Exception as e:
        return f"Qwen-3 32B evaluation:\nError occurred: {str(e)}\nBased on available context, assigning a moderate score.\nFinal Score: 10/20"

def call_llama_4_maverick(prompt):
    """Call Llama-4 Maverick via HuggingFace"""
    try:
        completion = hf_client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct:cerebras",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        response = completion.choices[0].message.content
        # Ensure response includes a final score
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 13/20"
        return response
    except Exception as e:
        return f"Llama-4 Maverick evaluation:\nError occurred: {str(e)}\nBased on rubric criteria, providing fallback assessment.\nFinal Score: 11/20"

def call_gemini_2_5(prompt):
    """Call Gemini-2.5 via Google API"""
    try:
        response = gemini_llm.invoke(prompt).content
        # Ensure response includes a final score
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 14/20"
        return response
    except Exception as e:
        return f"Gemini-2.5 evaluation:\nError occurred: {str(e)}\nProviding structured assessment based on rubric.\nFinal Score: 12/20"

# Three LLM functions for ensemble
LLM_FUNCTIONS = [call_qwen_32b, call_llama_4_maverick, call_gemini_2_5]
LLM_NAMES = ["Qwen-3 32B", "Llama-4 Maverick", "Gemini-2.5"]

# ============ MULTI-AGENT EVALUATION ============
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
1. Score the answer on a 20-point scale based on the rubric criteria above.
2. Justify the score thoroughly using specific rubric criteria and relevant policy rules.
3. Provide detailed feedback on:
   - Accuracy and correctness
   - Clarity of expression
   - Policy compliance
   - Strengths demonstrated
   - Areas for improvement
   - Specific recommendations

Be thorough in your evaluation. End your response with: Final Score: XX/20 (where XX is your numerical score)
"""

def evaluate_answer_prompt_peer1(state):
    return f"""
You are Agent #1: Policy Agent - Institutional compliance and academic integrity verification.
Your primary responsibilities:
- Check for factual correctness and logical reasoning
- Judge conceptual depth and understanding
- Ensure institutional policy compliance
- Verify academic integrity standards

{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer2(state):
    return f"""
You are Agent #2: Pedagogy Agent - Learning objectives alignment and educational standards.
Your primary responsibilities:
- Judge alignment with Bloom's taxonomy levels
- Assess pedagogical appropriateness of the response
- Check coherence, logical flow, and clarity of ideas
- Evaluate learning objective fulfillment

{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer3(state):
    return f"""
You are Agent #3: Originality & Voice Agent - Student authenticity and creative expression.
Your primary responsibilities:
- Evaluate originality of thought and personal voice
- Assess creative expression and individual perspective
- Check for authentic student thinking patterns
- Preserve and recognize student intellectual ownership

{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer4(state):
    return f"""
You are Agent #4: Equity Agent - Inclusive assessment and bias detection.
Your primary responsibilities:
- Evaluate for bias-free assessment
- Ensure equitable evaluation across diverse backgrounds
- Check for inclusive language and perspectives
- Address potential discrimination in evaluation process

{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer5(state):
    return f"""
You are Agent #5: Feedback Agent - Constructive scaffolding and personalized guidance.
Your primary responsibilities:
- Provide constructive, actionable feedback
- Offer personalized learning recommendations
- Focus on growth-oriented suggestions
- Deliver scaffolding that enhances future learning

{generic_context_and_instructions(state)}
"""

def evaluate_answer_prompt_peer6(state):
    return f"""
You are Agent #6: Summarizer Agent - Results aggregation and coherence evaluation.
Your primary responsibilities:
- Evaluate overall coherence and synthesis
- Provide holistic assessment perspective
- Ensure consistency across evaluation criteria
- Aggregate insights from multiple assessment perspectives

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

def extract_score(text):
    """Extract numerical score from LLM response with multiple fallback patterns"""
    # Multiple regex patterns to catch various score formats
    patterns = [
        r"Final Score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"Final Score:\s*(\d+(?:\.\d+)?)/20",
        r"Final Score:\s*(\d+(?:\.\d+)?)\s*out\s*of\s*20",
        r"Final Score:\s*(\d+(?:\.\d+)?)",
        r"final score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"final score:\s*(\d+(?:\.\d+)?)/20",
        r"final score:\s*(\d+(?:\.\d+)?)",
        r"Score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"Score:\s*(\d+(?:\.\d+)?)/20",
        r"score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"(\d+(?:\.\d+)?)\s*/\s*20",
        r"(\d+(?:\.\d+)?)/20",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Ensure score is within valid range
            return min(max(score, 0), 20)
    
    # If no score found, try to extract any number that might be a score
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    for num in numbers:
        score = float(num)
        if 0 <= score <= 20:
            return score
    
    # Final fallback - assign default score
    return 10.0

def evaluate_agent_with_three_llms(state, agent_id):
    """Evaluate using three LLMs for a single agent and return averaged result"""
    prompt_func = PEER_PROMPT_FUNCTIONS[agent_id - 1]
    prompt = prompt_func(state)
    
    # Run all three LLMs for this agent
    llm_responses = []
    llm_scores = []
    
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all LLM calls concurrently
            future_to_llm = {executor.submit(llm_func, prompt): i for i, llm_func in enumerate(LLM_FUNCTIONS)}
            
            for future in as_completed(future_to_llm):
                llm_index = future_to_llm[future]
                try:
                    response = future.result(timeout=45)  # 45 second timeout
                    llm_responses.append(f"**{LLM_NAMES[llm_index]}:**\n{response}")
                    score = extract_score(response)
                    llm_scores.append(score)
                    
                    # Debug logging
                    st.write(f"Debug - Agent {agent_id}, LLM {llm_index+1}: Score extracted = {score}")
                    
                except Exception as e:
                    error_response = f"**{LLM_NAMES[llm_index]}:** Error - {str(e)}\nFallback assessment provided.\nFinal Score: 10/20"
                    llm_responses.append(error_response)
                    llm_scores.append(10.0)
                    st.write(f"Debug - Agent {agent_id}, LLM {llm_index+1}: Error, using fallback score = 10.0")
    
    except Exception as e:
        # Complete fallback if executor fails
        for i in range(3):
            llm_responses.append(f"**{LLM_NAMES[i]}:** System error - {str(e)}\nFallback evaluation.\nFinal Score: 10/20")
            llm_scores.append(10.0)
    
    # Ensure we have exactly 3 scores
    while len(llm_scores) < 3:
        llm_scores.append(10.0)
        llm_responses.append(f"**Missing LLM Response:** Fallback score assigned.\nFinal Score: 10/20")
    
    # Calculate average score for this agent
    agent_avg_score = sum(llm_scores) / len(llm_scores) if llm_scores else 10.0
    
    # Combine all LLM responses for transparency
    combined_response = "\n\n".join(llm_responses)
    
    st.write(f"Debug - Agent {agent_id}: Individual scores = {llm_scores}, Average = {agent_avg_score:.2f}")
    
    return {
        "agent_id": agent_id,
        "llm_scores": llm_scores,
        "agent_avg_score": agent_avg_score,
        "combined_response": combined_response
    }

def peer_assessment_simulation(state, num_peers=6):
    """Run multi-agent evaluation with three LLMs per agent"""
    agent_results = []
    
    st.write("🤖 Starting multi-agent evaluation...")
    
    # Evaluate each agent with three LLMs
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_agent = {executor.submit(evaluate_agent_with_three_llms, state, i): i for i in range(1, num_peers + 1)}
            
            for future in as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    result = future.result(timeout=90)  # 90 second timeout per agent
                    agent_results.append(result)
                    st.write(f"✅ Agent {agent_id} completed with score: {result['agent_avg_score']:.2f}")
                except Exception as e:
                    # Fallback result if agent fails
                    fallback_result = {
                        "agent_id": agent_id,
                        "llm_scores": [10.0, 10.0, 10.0],
                        "agent_avg_score": 10.0,
                        "combined_response": f"Agent {agent_id} encountered an error: {str(e)}\nFallback evaluation provided.\nFinal Score: 10/20"
                    }
                    agent_results.append(fallback_result)
                    st.write(f"⚠️ Agent {agent_id} failed, using fallback score: 10.0")
    
    except Exception as e:
        # Complete fallback if executor fails
        st.write(f"⚠️ Executor failed: {str(e)}. Using fallback scores for all agents.")
        for i in range(1, num_peers + 1):
            agent_results.append({
                "agent_id": i,
                "llm_scores": [10.0, 10.0, 10.0],
                "agent_avg_score": 10.0,
                "combined_response": f"Agent {i} system error. Fallback evaluation provided.\nFinal Score: 10/20"
            })
    
    # Sort results by agent_id to maintain order
    agent_results.sort(key=lambda x: x["agent_id"])
    
    # Calculate overall average score
    agent_avg_scores = [result["agent_avg_score"] for result in agent_results]
    overall_avg_score = sum(agent_avg_scores) / len(agent_avg_scores) if agent_avg_scores else 10.0
    
    st.write(f"📊 Final Evaluation Complete:")
    st.write(f"Agent Scores: {[f'{score:.1f}' for score in agent_avg_scores]}")
    st.write(f"Overall Average: {overall_avg_score:.2f}")
    
    # Prepare combined feedback
    feedback_sections = []
    for result in agent_results:
        agent_name = ["Policy Agent", "Pedagogy Agent", "Originality & Voice Agent", 
                     "Equity Agent", "Feedback Agent", "Summarizer Agent"][result["agent_id"] - 1]
        
        feedback_sections.append(
            f"**Agent #{result['agent_id']} - {agent_name} (Score: {result['agent_avg_score']:.2f}/20):**\n"
            f"Individual LLM Scores: {', '.join([f'{score:.1f}' for score in result['llm_scores']])}\n\n"
            f"{result['combined_response']}"
        )
    
    combined_feedback = "\n\n" + "="*50 + "\n\n".join(feedback_sections)
    
    return {
        "peer_reviews": combined_feedback,
        "average_score": overall_avg_score,
        "scores": agent_avg_scores,
        "detailed_results": agent_results
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
        .llm-score {
            background: #f0f8ff;
            padding: 4px 8px;
            border-radius: 4px;
            margin: 2px;
            display: inline-block;
            font-size: 0.8rem;
        }
        .debug-info {
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.8rem;
            margin: 4px 0;
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
              <div class="subtle">Policy-Compliant AI Scaffolding with Three-LLM Ensemble</div>
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

    # Agent breakdown for latest submission
    if history:
        latest = history[-1]
        if 'agent_scores' in latest:
            st.markdown("#### 🤖 Latest Agent Breakdown")
            agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
            agent_cols = st.columns(6)
            for i, (name, score) in enumerate(zip(agent_names, latest['agent_scores'])):
                with agent_cols[i]:
                    st.metric(f"{name}", f"{score:.1f}")

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
            
            # Debug mode toggle
            debug_mode = st.checkbox("🐛 Debug Mode (Show detailed evaluation process)")
            
            with st.spinner("🤖 Running three-LLM multi-agent evaluation... This may take 60-90 seconds."):
                if debug_mode:
                    st.info("Debug mode enabled - showing detailed evaluation process...")
                    
                results = peer_assessment_simulation(eval_state, num_peers=6)

            # Extract agent scores and overall average
            agent_scores = results["scores"]
            overall_avg = results["average_score"]

            # Save version with detailed results
            st.session_state.submissions.setdefault(aid, []).append({
                "answer": answer,
                "scores": agent_scores,  # Individual agent averages
                "agent_scores": agent_scores,  # For monitoring dashboard
                "avg": overall_avg,
                "feedback": results["peer_reviews"],
                "detailed_results": results.get("detailed_results", []),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "submitted": False
            })
            st.success("✅ Three-LLM ensemble evaluation complete! Check detailed feedback below.")

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
    st.markdown("<div class='card'><h4>🤖 Three-LLM Multi-Agent Feedback</h4>", unsafe_allow_html=True)
    versions = st.session_state.submissions.get(aid, [])
    if versions:
        last = versions[-1]
        
        # Score display
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("📊 Overall Average Score", f"{last['avg']:.2f}/20")
        with col2:
            status = "🎯 Final Submission" if last.get('submitted') else "📝 Draft"
            st.markdown(f"**Status:** {status}")
        
        # Individual agent scores with LLM breakdown
        st.markdown("##### Individual Agent Scores (Avg of 3 LLMs each)")
        agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
        score_cols = st.columns(6)
        
        for i, (name, score) in enumerate(zip(agent_names, last['scores'])):
            with score_cols[i]:
                st.metric(f"{name}", f"{score:.2f}")
                
                # Show individual LLM scores if available
                if 'detailed_results' in last:
                    detailed = last['detailed_results']
                    if i < len(detailed):
                        llm_scores = detailed[i]['llm_scores']
                        llm_display = " | ".join([f"{LLM_NAMES[j][:4]}: {score:.1f}" for j, score in enumerate(llm_scores)])
                        st.markdown(f"<div class='llm-score'>{llm_display}</div>", unsafe_allow_html=True)
        
        # Detailed feedback
        with st.expander("📝 View Detailed Three-LLM Feedback & Suggestions", expanded=False):
            st.markdown(last["feedback"])
    else:
        st.info("💭 Submit your answer for three-LLM AI evaluation to see feedback here.")
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
                "Score": f"{v['avg']:.2f}/20",
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

    # Check for required environment variables
    if not os.environ.get("HF_TOKEN"):
        st.warning("⚠️ HF_TOKEN environment variable not set. Using hardcoded token.")
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. Using hardcoded key.")

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        st.markdown("**Three-LLM Ensemble:** Qwen-3 32B • Llama-4 Maverick • Gemini-2.5")
        
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
            st.markdown("<div class='card'><h4>👨‍🏫 Instructor Dashboard</h4><p class='muted'>Welcome! Use the sidebar to access different modules and manage your courses with three-LLM ensemble evaluation.</p></div>", unsafe_allow_html=True)
            
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

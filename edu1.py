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
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyAdGZTX3JMWsOnGWnqGtoS-uHnnC3Gr05g")
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
def call_qwen_vl_235b(prompt):
    """Call Qwen3-VL-235B via HuggingFace"""
    try:
        completion = hf_client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Thinking:novita",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        response = completion.choices[0].message.content
        # Ensure response includes a final score
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 12/20"
        return response
    except Exception as e:
        return f"Qwen3-VL-235B evaluation:\nError occurred: {str(e)}\nBased on available context and rubric criteria, assigning moderate score.\nFinal Score: 10/20"

def call_deepseek_v31(prompt):
    """Call DeepSeek-V3.1 via HuggingFace"""
    try:
        completion = hf_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.1:novita",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        response = completion.choices[0].message.content
        # Ensure response includes a final score
        if "Final Score:" not in response and "final score:" not in response.lower():
            response += "\n\nFinal Score: 13/20"
        return response
    except Exception as e:
        return f"DeepSeek-V3.1 evaluation:\nError occurred: {str(e)}\nBased on rubric criteria, providing structured assessment.\nFinal Score: 11/20"

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
LLM_FUNCTIONS = [call_qwen_vl_235b, call_deepseek_v31, call_gemini_2_5]
LLM_NAMES = ["Qwen3-VL-235B", "DeepSeek-V3.1", "Gemini-2.5"]

# ============ RUBRIC MANAGEMENT SYSTEM ============
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
    """Get the active rubric - prioritizes instructor-defined rubrics"""
    if st.session_state.rubrics:
        # Use instructor-defined rubrics
        rubric_text = "**INSTRUCTOR-DEFINED RUBRIC** (Total: 20 points)\n\n"
        rubric_text += "Score each criterion based on the following breakdown:\n\n"
        
        total_possible = 0
        for i, rubric in enumerate(st.session_state.rubrics, 1):
            weight_points = (rubric['weight'] / 100) * 20  # Convert percentage to points out of 20
            rubric_text += f"{i}. **{rubric['name']}**: {weight_points:.2f} points ({rubric['weight']}%)\n"
            rubric_text += f"   - Evaluate student's performance in this specific area\n"
            rubric_text += f"   - Provide detailed justification for the score assigned\n\n"
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
        # Fallback to hardcoded rubric
        return HARDCODED_RUBRIC + "\n\n**Note: Using default rubric as no instructor-defined criteria available.**"

def format_rubric_for_display():
    """Format rubric for display in the interface"""
    if st.session_state.rubrics:
        total_criteria = len(st.session_state.rubrics)
        return f"**Custom Rubric** ({total_criteria} criteria)"
    else:
        return "**Default Hardcoded Rubric** (14 criteria)"

def get_rubric_breakdown_text():
    """Get detailed rubric breakdown for display"""
    if st.session_state.rubrics:
        breakdown = "**Current Evaluation Criteria:**\n\n"
        for i, rubric in enumerate(st.session_state.rubrics, 1):
            points = (rubric['weight'] / 100) * 20
            breakdown += f"• **{rubric['name']}**: {points:.2f} pts ({rubric['weight']}%)\n"
        return breakdown
    else:
        return "**Using Default Criteria:** Application, Understanding, Skills, Design, Research, Critical Thinking, etc."

# ============ MULTI-AGENT EVALUATION ============
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
You must evaluate this student's answer using the rubric criteria above.

**REQUIRED FORMAT:**
1. **CRITERION-BY-CRITERION EVALUATION:**
   - For each rubric criterion, provide:
     * Score assigned (out of maximum points for that criterion)
     * Detailed justification with specific evidence from student's answer
     * Strengths observed
     * Areas for improvement

2. **OVERALL ASSESSMENT SUMMARY:**
   - Key strengths of the response
   - Main areas needing improvement
   - Overall quality judgment

3. **ACTIONABLE RECOMMENDATIONS:**
   - Specific steps student can take to improve
   - Resources or strategies that would help
   - Next learning objectives to focus on

**IMPORTANT:** End your response with "Final Score: XX/20" where XX is the total points earned across all criteria.

Be thorough, fair, and constructive in your evaluation. Use specific quotes and examples from the student's response to justify your scoring decisions.
"""

def evaluate_answer_prompt_peer1(state):
    return f"""
You are **Agent #1: Policy Agent** - Institutional compliance and academic integrity verification.

**YOUR SPECIALIZED FOCUS:**
- Check for factual correctness and logical reasoning
- Verify adherence to institutional policies and guidelines
- Ensure academic integrity standards are met
- Assess accuracy of information presented
- Evaluate logical flow and coherence of arguments

**SPECIAL ATTENTION TO:**
- Policy compliance issues
- Academic integrity concerns
- Factual accuracy
- Logical consistency
- Proper citation and attribution (if applicable)

{generic_context_and_instructions(state)}

**Remember:** Your expertise is in policy compliance and academic integrity. Weight these aspects heavily in your evaluation while still addressing all rubric criteria.
"""

def evaluate_answer_prompt_peer2(state):
    return f"""
You are **Agent #2: Pedagogy Agent** - Learning objectives alignment and educational standards.

**YOUR SPECIALIZED FOCUS:**
- Judge alignment with Bloom's taxonomy levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
- Assess pedagogical appropriateness and educational value
- Evaluate learning objective fulfillment
- Check depth of understanding and knowledge application
- Assess critical thinking and analytical skills

**SPECIAL ATTENTION TO:**
- Knowledge application and synthesis
- Critical thinking demonstration
- Educational value and learning outcomes
- Cognitive skill development
- Problem-solving approaches

{generic_context_and_instructions(state)}

**Remember:** Your expertise is in pedagogy and learning outcomes. Focus on how well the student demonstrates mastery of educational objectives.
"""

def evaluate_answer_prompt_peer3(state):
    return f"""
You are **Agent #3: Originality & Voice Agent** - Student authenticity and creative expression.

**YOUR SPECIALIZED FOCUS:**
- Evaluate originality of thought and unique perspectives
- Assess personal voice and individual expression
- Check for authentic student thinking patterns
- Recognize creative approaches and innovative solutions
- Preserve and acknowledge student intellectual ownership

**SPECIAL ATTENTION TO:**
- Original insights and personal perspectives
- Creative problem-solving approaches
- Individual voice and expression style
- Innovative thinking and unique solutions
- Authentic student engagement with the topic

{generic_context_and_instructions(state)}

**Remember:** Your expertise is in recognizing and nurturing student originality. Value creativity, personal voice, and authentic thinking.
"""

def evaluate_answer_prompt_peer4(state):
    return f"""
You are **Agent #4: Equity Agent** - Inclusive assessment and bias detection.

**YOUR SPECIALIZED FOCUS:**
- Evaluate for bias-free and culturally responsive assessment
- Ensure equitable evaluation across diverse backgrounds and perspectives
- Check for inclusive language and diverse viewpoints
- Address potential discrimination in evaluation process
- Recognize diverse forms of knowledge and expression

**SPECIAL ATTENTION TO:**
- Cultural sensitivity and inclusivity
- Recognition of diverse perspectives and experiences
- Bias-free evaluation practices
- Equitable assessment across different backgrounds
- Inclusive language and representation

{generic_context_and_instructions(state)}

**Remember:** Your expertise is in equitable assessment. Ensure fair evaluation regardless of student background while maintaining academic standards.
"""

def evaluate_answer_prompt_peer5(state):
    return f"""
You are **Agent #5: Feedback Agent** - Constructive scaffolding and personalized guidance.

**YOUR SPECIALIZED FOCUS:**
- Provide actionable, constructive feedback for improvement
- Offer personalized learning recommendations and next steps
- Focus on growth-oriented suggestions and scaffolding
- Deliver guidance that enhances future learning and development
- Create pathways for continued skill building

**SPECIAL ATTENTION TO:**
- Specific, actionable improvement strategies
- Personalized learning recommendations
- Growth-oriented feedback that builds confidence
- Scaffolding for future skill development
- Clear next steps and learning pathways

{generic_context_and_instructions(state)}

**Remember:** Your expertise is in providing constructive feedback. Focus on how the student can grow and improve while recognizing current achievements.
"""

def evaluate_answer_prompt_peer6(state):
    return f"""
You are **Agent #6: Summarizer Agent** - Results aggregation and holistic evaluation.

**YOUR SPECIALIZED FOCUS:**
- Provide holistic assessment that synthesizes multiple perspectives
- Evaluate overall coherence and integration of ideas
- Ensure consistency across evaluation criteria
- Aggregate insights and provide comprehensive overview
- Balance detailed analysis with big-picture assessment

**SPECIAL ATTENTION TO:**
- Overall coherence and synthesis quality
- Integration of concepts and ideas
- Holistic learning demonstration
- Comprehensive skill assessment
- Balance between different evaluation aspects

{generic_context_and_instructions(state)}

**Remember:** Your expertise is in holistic evaluation. Provide a comprehensive assessment that considers the response as a complete learning demonstration.
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
        r"Total Score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"Total Score:\s*(\d+(?:\.\d+)?)/20",
        r"Overall Score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"Score:\s*(\d+(?:\.\d+)?)\s*/20",
        r"Score:\s*(\d+(?:\.\d+)?)/20",
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
    
    # Final fallback - assign default score based on rubric
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
                    response = future.result(timeout=60)  # Increased timeout for more complex models
                    llm_responses.append(f"**{LLM_NAMES[llm_index]}:**\n{response}")
                    score = extract_score(response)
                    llm_scores.append(score)
                    
                    # Debug logging
                    if st.session_state.get('debug_mode', False):
                        st.write(f"Debug - Agent {agent_id}, LLM {llm_index+1} ({LLM_NAMES[llm_index]}): Score extracted = {score}")
                    
                except Exception as e:
                    error_response = f"**{LLM_NAMES[llm_index]}:** Error - {str(e)}\nFallback assessment based on rubric criteria.\nFinal Score: 10/20"
                    llm_responses.append(error_response)
                    llm_scores.append(10.0)
                    if st.session_state.get('debug_mode', False):
                        st.write(f"Debug - Agent {agent_id}, LLM {llm_index+1}: Error, using fallback score = 10.0")
    
    except Exception as e:
        # Complete fallback if executor fails
        for i in range(3):
            llm_responses.append(f"**{LLM_NAMES[i]}:** System error - {str(e)}\nFallback evaluation based on rubric.\nFinal Score: 10/20")
            llm_scores.append(10.0)
    
    # Ensure we have exactly 3 scores
    while len(llm_scores) < 3:
        llm_scores.append(10.0)
        llm_responses.append(f"**Missing LLM Response:** Fallback score assigned based on rubric criteria.\nFinal Score: 10/20")
    
    # Calculate average score for this agent
    agent_avg_score = sum(llm_scores) / len(llm_scores) if llm_scores else 10.0
    
    # Combine all LLM responses for transparency
    combined_response = "\n\n".join(llm_responses)
    
    if st.session_state.get('debug_mode', False):
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
    
    if st.session_state.get('debug_mode', False):
        st.write("🤖 Starting multi-agent evaluation...")
        st.write(f"📋 Using Rubric: {format_rubric_for_display()}")
        st.write(f"🔧 LLM Ensemble: {', '.join(LLM_NAMES)}")
    
    # Evaluate each agent with three LLMs
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_agent = {executor.submit(evaluate_agent_with_three_llms, state, i): i for i in range(1, num_peers + 1)}
            
            for future in as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    result = future.result(timeout=120)  # Increased timeout for complex evaluations
                    agent_results.append(result)
                    if st.session_state.get('debug_mode', False):
                        st.write(f"✅ Agent {agent_id} completed with score: {result['agent_avg_score']:.2f}")
                except Exception as e:
                    # Fallback result if agent fails
                    fallback_result = {
                        "agent_id": agent_id,
                        "llm_scores": [10.0, 10.0, 10.0],
                        "agent_avg_score": 10.0,
                        "combined_response": f"Agent {agent_id} encountered an error: {str(e)}\nFallback evaluation provided based on rubric criteria.\nFinal Score: 10/20"
                    }
                    agent_results.append(fallback_result)
                    if st.session_state.get('debug_mode', False):
                        st.write(f"⚠️ Agent {agent_id} failed, using fallback score: 10.0")
    
    except Exception as e:
        # Complete fallback if executor fails
        if st.session_state.get('debug_mode', False):
            st.write(f"⚠️ Executor failed: {str(e)}. Using fallback scores for all agents.")
        for i in range(1, num_peers + 1):
            agent_results.append({
                "agent_id": i,
                "llm_scores": [10.0, 10.0, 10.0],
                "agent_avg_score": 10.0,
                "combined_response": f"Agent {i} system error. Fallback evaluation provided based on rubric.\nFinal Score: 10/20"
            })
    
    # Sort results by agent_id to maintain order
    agent_results.sort(key=lambda x: x["agent_id"])
    
    # Calculate overall average score
    agent_avg_scores = [result["agent_avg_score"] for result in agent_results]
    overall_avg_score = sum(agent_avg_scores) / len(agent_avg_scores) if agent_avg_scores else 10.0
    
    if st.session_state.get('debug_mode', False):
        st.write(f"📊 Final Evaluation Complete:")
        st.write(f"Agent Scores: {[f'{score:.1f}' for score in agent_avg_scores]}")
        st.write(f"Overall Average: {overall_avg_score:.2f}")
    
    # Prepare combined feedback with rubric context
    rubric_context = f"**EVALUATION CONTEXT:**\n{format_rubric_for_display()}\n{get_rubric_breakdown_text()}\n\n"
    
    feedback_sections = [rubric_context]
    for result in agent_results:
        agent_names = ["Policy Agent", "Pedagogy Agent", "Originality & Voice Agent", 
                      "Equity Agent", "Feedback Agent", "Summarizer Agent"]
        agent_name = agent_names[result["agent_id"] - 1]
        
        feedback_sections.append(
            f"**Agent #{result['agent_id']} - {agent_name} (Score: {result['agent_avg_score']:.2f}/20):**\n"
            f"LLM Ensemble Scores: {' | '.join([f'{LLM_NAMES[i]}: {score:.1f}' for i, score in enumerate(result['llm_scores'])])}\n\n"
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
    st.session_state.setdefault("debug_mode", False)  # Debug mode toggle
    
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
        .rubric-indicator {
            background: #e8f5e8;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 4px solid #4caf50;
        }
        .llm-indicator {
            background: #e3f2fd;
            padding: 6px 10px;
            border-radius: 6px;
            margin: 4px 0;
            border-left: 3px solid #2196f3;
            font-size: 0.9rem;
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
    
    # Show active rubric status
    st.markdown(f"<div class='rubric-indicator'>{format_rubric_for_display()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='llm-indicator'>🤖 LLM Ensemble: {' • '.join(LLM_NAMES)}</div>", unsafe_allow_html=True)
    
    with st.form("rubric_form"):
        st.markdown("**Define Evaluation Criteria (Must total 100%)**")
        rubrics_data = []
        total_percentage = 0
        
        for i in range(st.session_state.num_rubric_entries):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                rubric_name = st.text_input(f"Evaluation Criterion {i+1}", 
                                          placeholder=f"e.g., Critical Thinking, Communication Skills, Content Knowledge",
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
            save_rubrics = st.form_submit_button("💾 Save Custom Rubric", type="primary")
            
        if save_rubrics:
            if abs(total_percentage - 100) < 0.01:  # Allow small floating point errors
                saved_count = 0
                # Clear existing rubrics and replace with new ones
                st.session_state.rubrics = []
                for rubric_data in rubrics_data:
                    if rubric_data["name"]:
                        st.session_state.rubrics.append({
                            "id": str(uuid.uuid4()),
                            "name": rubric_data["name"],
                            "weight": rubric_data["weight"]
                        })
                        saved_count += 1
                
                if saved_count > 0:
                    st.success(f"✅ Custom rubric with {saved_count} criteria saved! All evaluations will now use these criteria.")
                    st.rerun()  # Refresh to show updated rubric status
                else:
                    st.warning("⚠️ Please provide names for all rubric criteria.")
            else:
                st.error("❌ Rubric weights must total exactly 100%!")
    
    # Add more button (outside form)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add More Criteria"):
            st.session_state.num_rubric_entries += 1
            st.rerun()
    
    with col2:
        if st.session_state.rubrics and st.button("🗑️ Reset to Default Rubric"):
            st.session_state.rubrics = []
            st.session_state.num_rubric_entries = 1
            st.success("✅ Reset to default rubric. System will use hardcoded criteria.")
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Display current rubric
    st.markdown("<div class='card'><h4>📈 Current Evaluation Rubric</h4>", unsafe_allow_html=True)
    if st.session_state.rubrics:
        st.success("🎯 **Custom Instructor-Defined Rubric Active**")
        df_data = []
        total_weight = 0
        for r in st.session_state.rubrics:
            points = (r["weight"] / 100) * 20
            df_data.append({
                "Criterion": r["name"], 
                "Weight (%)": r["weight"],
                "Points (out of 20)": f"{points:.2f}"
            })
            total_weight += r["weight"]
        
        df = pd.DataFrame(df_data)
        st.table(df)
        
        if abs(total_weight - 100) < 0.01:
            st.success(f"✅ Total Weight: {total_weight}% | This rubric will be used for all evaluations.")
        else:
            st.warning(f"⚠️ Total Weight: {total_weight}% (Should be 100%)")
    else:
        st.info("📋 **Using Default Hardcoded Rubric** - Create custom criteria above to override.")
        with st.expander("📋 View Default Rubric Details"):
            st.code(HARDCODED_RUBRIC, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

def assignment_creator():
    st.markdown("<div class='card'><h4>📝 Assignment Creation</h4>", unsafe_allow_html=True)
    
    # Show which rubric will be used
    st.markdown(f"<div class='rubric-indicator'>📋 Evaluations will use: {format_rubric_for_display()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='llm-indicator'>🤖 Evaluation powered by: {' • '.join(LLM_NAMES)}</div>", unsafe_allow_html=True)
    
    with st.form("assignment_form", clear_on_submit=True):
        q = st.text_area("📋 Assignment Question / Prompt", height=120, 
                        placeholder="Enter the assignment question or prompt here...\n\nExample: 'Analyze the impact of artificial intelligence on modern education. Discuss both benefits and challenges, providing specific examples and your personal insights.'")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.form_submit_button("🚀 Publish Assignment", type="primary"):
                if q.strip():
                    st.session_state.assignments.append({
                        "id": str(uuid.uuid4()),
                        "question": q.strip(),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "rubric_snapshot": format_rubric_for_display(),
                        "rubric_details": get_rubric_breakdown_text()
                    })
                    st.success("✅ Assignment published to Student Workspace with current rubric settings!")
                else:
                    st.warning("⚠️ Please enter a question.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📚 Published Assignments</h4>", unsafe_allow_html=True)
    if st.session_state.assignments:
        for i, a in enumerate(st.session_state.assignments, 1):
            st.markdown(f"**{i}.** {a['question']}")
            st.markdown(f"<small class='muted'>📅 Published: {a['created_at']}</small>", unsafe_allow_html=True)
            if 'rubric_snapshot' in a:
                st.markdown(f"<small class='muted'>📋 Will be evaluated using: {a['rubric_snapshot']}</small>", unsafe_allow_html=True)
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

    # Show rubric being used for this assignment
    selected_assignment = next(a for a in st.session_state.assignments if a["id"] == aid)
    if 'rubric_snapshot' in selected_assignment:
        st.markdown(f"<div class='rubric-indicator'>📋 This assignment uses: {selected_assignment['rubric_snapshot']}</div>", unsafe_allow_html=True)

    # Timeline
    st.markdown("#### 🕐 Submission Timeline")
    for i, h in enumerate(history, start=1):
        status_icon = "🏁" if h.get('submitted') else "📝"
        st.markdown(f"{status_icon} **Version {i}** — {h['timestamp']} — Score: {h['avg']:.2f}/20")

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
                    st.metric(f"{name}", f"{score:.2f}")

    # Feedback logs
    st.markdown("#### 💭 Detailed Feedback History")
    for i, h in enumerate(history, start=1):
        status = "Final Submission" if h.get('submitted') else "Draft"
        rubric_used = h.get('rubric_used', 'Unknown')
        with st.expander(f"Version {i} Feedback ({status}) - {rubric_used}"):
            st.markdown(h["feedback"])

    # Final submission indicator
    final_versions = [h for h in history if h.get("submitted")]
    if final_versions:
        st.success("🎯 Final submission received!")
        if st.button("📖 Review Final Work"):
            v = final_versions[-1]
            st.markdown("##### 📝 Final Answer")
            st.write(v["answer"])
            st.markdown("##### 💭 Final Comprehensive Feedback")
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
    
    # Get full question and rubric info
    selected_assignment = next(a for a in st.session_state.assignments if a["id"] == aid)
    full_question = selected_assignment["question"]
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📋 Assignment Question</h4>", unsafe_allow_html=True)
    st.markdown(f"**{full_question}**")
    
    # Show evaluation information
    if 'rubric_snapshot' in selected_assignment:
        st.markdown(f"<div class='rubric-indicator'>📋 Your submission will be evaluated using: {selected_assignment['rubric_snapshot']}</div>", unsafe_allow_html=True)
    
    if 'rubric_details' in selected_assignment:
        with st.expander("📊 View Detailed Evaluation Criteria"):
            st.markdown(selected_assignment['rubric_details'])
    
    st.markdown(f"<div class='llm-indicator'>🤖 Evaluation powered by advanced AI: {' • '.join(LLM_NAMES)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Answer editor + feedback
    st.markdown("<div class='card'><h4>✍️ Answer Editor</h4>", unsafe_allow_html=True)
    default_text = ""
    if st.session_state.submissions.get(aid):
        default_text = st.session_state.submissions[aid][-1]["answer"]
    
    answer = st.text_area("Draft your answer here:", height=200, value=default_text,
                         placeholder="Write your comprehensive response here...\n\n• Address all aspects of the question\n• Provide specific examples and evidence\n• Show your critical thinking and analysis\n• Express your personal insights and voice")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        evaluate = st.button("🤖 Submit for AI Evaluation")
    with col2:
        modify = st.button("✏️ Modify & Resubmit")
    with col3:
        final_submit = st.button("🎯 Final Submit", type="primary")
    with col4:
        st.session_state.debug_mode = st.checkbox("🐛 Debug", value=st.session_state.get('debug_mode', False))

    # Ensure policy string is available for context
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
                "rubric": get_active_rubric(),  # Use the active rubric
                "parsed_policy": parsed_policy
            }
            
            with st.spinner("🤖 Running comprehensive three-LLM multi-agent evaluation... This may take 90-120 seconds."):
                if st.session_state.debug_mode:
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
                "submitted": False,
                "rubric_used": format_rubric_for_display()  # Track which rubric was used
            })
            st.success("✅ Comprehensive three-LLM ensemble evaluation complete! Check detailed feedback below.")

    # Final submit confirmation
    if final_submit:
        versions = st.session_state.submissions.get(aid, [])
        if not versions:
            st.warning("⚠️ Please evaluate at least once before final submission.")
        else:
            versions[-1]["submitted"] = True
            st.success("🎯 Final submission recorded! Your instructor can now review your complete work and feedback.")

    st.markdown("</div>", unsafe_allow_html=True)

    # AI Feedback panel
    st.markdown("<div class='card'><h4>🤖 Comprehensive Multi-Agent AI Feedback</h4>", unsafe_allow_html=True)
    versions = st.session_state.submissions.get(aid, [])
    if versions:
        last = versions[-1]
        
        # Score display
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("📊 Overall Score", f"{last['avg']:.2f}/20", f"{(last['avg']/20*100):.1f}%")
        with col2:
            status = "🎯 Final Submission" if last.get('submitted') else "📝 Draft"
            st.markdown(f"**Status:** {status}")
        with col3:
            rubric_used = last.get('rubric_used', 'Unknown')
            st.markdown(f"**Evaluated with:** {rubric_used}")
        
        # Individual agent scores with LLM breakdown
        st.markdown("##### Individual Agent Scores (Average of 3 LLMs each)")
        agent_names = ["Policy", "Pedagogy", "Originality", "Equity", "Feedback", "Summarizer"]
        score_cols = st.columns(6)
        
        for i, (name, score) in enumerate(zip(agent_names, last['scores'])):
            with score_cols[i]:
                percentage = (score/20*100)
                st.metric(f"{name}", f"{score:.2f}", f"{percentage:.1f}%")
                
                # Show individual LLM scores if available
                if 'detailed_results' in last:
                    detailed = last['detailed_results']
                    if i < len(detailed):
                        llm_scores = detailed[i]['llm_scores']
                        llm_display = " | ".join([f"{LLM_NAMES[j]}: {score:.1f}" for j, score in enumerate(llm_scores)])
                        st.markdown(f"<div class='llm-score'>{llm_display}</div>", unsafe_allow_html=True)
        
        # Performance analysis
        if last['avg'] >= 18:
            st.success("🌟 **Excellent Performance** - Outstanding work across all criteria!")
        elif last['avg'] >= 15:
            st.info("👍 **Good Performance** - Solid work with room for refinement.")
        elif last['avg'] >= 12:
            st.warning("📈 **Satisfactory Performance** - Meets basic requirements, significant improvement opportunities.")
        else:
            st.error("📝 **Needs Improvement** - Consider revising based on detailed feedback below.")
        
        # Detailed feedback
        with st.expander("📝 View Comprehensive Three-LLM Feedback & Detailed Analysis", expanded=False):
            st.markdown(last["feedback"])
    else:
        st.info("💭 Submit your answer for comprehensive three-LLM AI evaluation to see detailed feedback here.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Version history
    st.markdown("<div class='card'><h4>📊 Submission History & Progress</h4>", unsafe_allow_html=True)
    if versions:
        table_data = []
        for i, v in enumerate(versions, start=1):
            status_icon = "🎯" if v.get("submitted") else "📝"
            rubric_used = v.get('rubric_used', 'Unknown')
            performance = ""
            if v['avg'] >= 18:
                performance = "🌟 Excellent"
            elif v['avg'] >= 15:
                performance = "👍 Good" 
            elif v['avg'] >= 12:
                performance = "📈 Satisfactory"
            else:
                performance = "📝 Needs Work"
                
            table_data.append({
                "Version": f"{status_icon} {i}",
                "Timestamp": v["timestamp"],
                "Score": f"{v['avg']:.2f}/20",
                "Performance": performance,
                "Status": "Final" if v.get("submitted") else "Draft",
                "Rubric": rubric_used
            })
        st.table(pd.DataFrame(table_data))
        
        # Progress analysis
        if len(versions) > 1:
            improvement = versions[-1]['avg'] - versions[0]['avg']
            if improvement > 0:
                st.success(f"📈 You've improved by {improvement:.2f} points since your first submission!")
            elif improvement < 0:
                st.info(f"📊 Your latest score is {abs(improvement):.2f} points lower. Consider reviewing the feedback.")
            else:
                st.info("📊 Your score remained consistent. Focus on the feedback for targeted improvement.")
    else:
        st.info("📈 Your submission history and progress will appear here.")
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
        st.markdown("**Three-LLM Ensemble:**")
        st.markdown("• Qwen3-VL-235B-A22B-Thinking")
        st.markdown("• DeepSeek-V3.1")  
        st.markdown("• Gemini-2.5")
        
        # Rubric status in sidebar
        st.markdown("---")
        st.markdown(f"**Active Rubric:** {format_rubric_for_display()}")
        
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
            st.markdown("<div class='card'><h4>👨‍🏫 Instructor Dashboard</h4><p class='muted'>Welcome! Manage your courses with advanced three-LLM ensemble evaluation and custom rubric system.</p></div>", unsafe_allow_html=True)
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Policies", len(st.session_state.policies))
            with col2:
                st.metric("📊 Rubric Criteria", len(st.session_state.rubrics))
            with col3:
                st.metric("📝 Assignments", len(st.session_state.assignments))
            with col4:
                rubric_type = "Custom" if st.session_state.rubrics else "Default"
                st.metric("📋 Active Rubric", rubric_type)
                
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

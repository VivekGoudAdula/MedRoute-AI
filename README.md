<<<<<<< HEAD
# 🏥 MedRoute AI: Healthcare Triage Simulator

An **OpenEnv-compliant reinforcement learning environment** that simulates real-world medical triage decision-making in rural and low-resource settings.

---

## 🌟 Problem Statement

In many developing regions, patients lack access to immediate medical expertise. As a result:

* Critical conditions are often **underestimated**, causing life-threatening delays.
* Mild conditions are frequently **over-escalated**, overloading hospitals unnecessarily.

There is currently **no standardized environment** to train AI agents on:

* asking clinically relevant questions
* identifying urgency levels
* making safe and cost-aware treatment decisions

**MedRoute AI bridges this gap** by providing a structured simulation environment for training and evaluating intelligent triage agents.

---

## 🎯 Objective

To build a **realistic, interactive environment** where an AI agent learns to:

* progressively gather medical information
* classify patient urgency
* recommend appropriate treatment pathways

using the OpenEnv interface (`step()`, `reset()`, `state()`).

---

## ⚙️ Environment Design

The system follows a **state-driven interaction loop**:

### 🔹 Observation (State)

* Initial symptoms
* Revealed symptoms (based on agent queries)
* Interaction history

### 🔹 Action Space

The agent can perform:

* `ASK`: Ask a follow-up medical question
* `CLASSIFY_URGENCY`: Predict severity (`low`, `medium`, `high`)
* `DECIDE_TREATMENT`: Recommend action (`home`, `clinic`, `hospital`, `emergency`)

### 🔹 Environment Behavior

* Reveals hidden symptoms dynamically
* Tracks question history and repetition
* Applies reward shaping based on performance

---

## 📋 Task Design

### 🟢 Task 1: Urgency Classification (Easy)

* Predict patient severity level
* **Reward:**

  * Correct → `1.0`
  * Adjacent → `0.5`
  * Incorrect → `0.0`

---

### 🟡 Task 2: Progressive Diagnosis (Medium)

* Ask relevant, non-redundant medical questions

* Discover hidden symptoms

* **Reward:**

  * Highly relevant → `+0.3`
  * Relevant → `+0.1`
  * Redundant → `-0.2`
  * Irrelevant → `-0.2`

---

### 🔴 Task 3: Treatment Decision (Hard)

* Recommend:
  `home`, `clinic`, `hospital`, `emergency`

* **Reward:**

  * Correct → `1.0`
  * Over-escalation → `0.5`
  * Underestimation (dangerous) → `0.0`

---

## 🧠 Reward Design (Key Innovation)

The reward function is designed to encourage:

* ✅ **Clinical correctness**
* ⚡ **Efficiency (fewer steps)**
* 🚨 **Early detection of critical cases**

### Reward Components:

* Step-wise reward for relevant questioning
* Penalty for repetition and inefficiency
* Bonus for early correct decisions
* Heavy penalty for unsafe decisions

This ensures the agent learns **safe, efficient, and realistic triage behavior**.

---

## 🏗️ Project Structure

```
medroute-ai/
│
├── env/
│   ├── environment.py   # Core environment logic
│   ├── models.py        # Pydantic models (OpenEnv spec)
│   ├── tasks.py         # Simulated patient cases
│   ├── grader.py        # Deterministic scoring
│
├── inference.py         # Baseline agent runner
├── openenv.yaml         # Environment metadata
├── Dockerfile           # Container setup
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Execution

### 📦 Installation

```bash
git clone <repo-url>
cd medroute-ai
pip install -r requirements.txt
```

---

### 🔐 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

---

### ▶️ Run Inference

```bash
python inference.py
```

---

## 🐳 Docker Execution

```bash
docker build -t medroute-ai .
docker run -e OPENAI_API_KEY=your_key medroute-ai
```

---

## 📊 Sample Output

```
[START]
Patient initial symptoms: ['sudden confusion', 'slurred speech']

[STEP] 1
Action: ASK -> Are you experiencing weakness in your limbs?

[STEP] 2
Action: CLASSIFY_URGENCY -> high

[STEP] 3
Action: DECIDE_TREATMENT -> emergency

[END]
```

---

## 🌍 Real-World Impact

MedRoute AI addresses a critical global challenge:

* Improves **early detection of life-threatening conditions**
* Reduces **unnecessary hospital burden**
* Enables **AI training in low-resource healthcare scenarios**

This system has potential applications in:

* rural telemedicine
* AI-assisted triage systems
* emergency response optimization

---

## 📌 Constraints

* Runs within **8GB RAM / 2 vCPU**
* Fully deterministic grading
* No external APIs required beyond inference

---

## 🧠 Why This Matters

MedRoute AI is not just a simulation — it is a **training ground for safer AI decision-making in healthcare**, where mistakes have real-world consequences.

By combining structured environments with intelligent reward design, it pushes forward the development of **trustworthy AI systems for critical domains**.
=======
---
title: Medroute Ai
emoji: 🏃
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: AI-powered healthcare triage simulation environment -OpenEnv
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 53d4059cde7661f9dab51e465dd07311725e0aaa

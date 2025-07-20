# InsuraLQ – Insurance Claim Legitimacy Analyzer (GPT-Powered)

**InsuraLQ** is an AI-driven insurance claim assessment tool that uses OpenAI's GPT models to evaluate the legitimacy of insurance claims through structured analysis and fraud detection metrics.

---

## Features

- **GPT-Powered Feature Extraction**  
  Uses GPT-4 to extract structured data points from raw claim descriptions, including timelines, evidence, and narrative consistency.

- **Fraud Indicator Detection**  
  Checks for predefined fraud indicators across categories like narrative consistency, timing patterns, behavioral flags, and more.

- **Risk Scoring System**  
  Calculates a dynamic fraud risk score (0–100) and assigns a risk level (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

- **Professional Explanation Generation**  
  Auto-generates a concise professional explanation for why a claim is flagged, suitable for insurance adjusters.

- **Actionable Recommendations**  
  Provides recommendations based on the detected risk level and specific fraud indicators.

- **CLI-Ready**  
  Comes with an interactive command-line interface for immediate use.

---

## System Requirements

- Python 3.7+
- OpenAI Python SDK

---

## Setup

1. Install dependencies:

   pip install openai
Set your OpenAI API key via environment variable or directly in code:

export OPENAI_API_KEY='your-openai-api-keHow It Works
Input the full insurance claim text via the CLI or integrate via API.

The model extracts features and scans for red flags using GPT-4.

Detected fraud indicators contribute to a weighted risk score.

Generates:

Risk assessment (score + level)

Detailed professional explanation

Recommendations for next steps

Usage
Run the tool directly:

bash
Copy
Edit
python insuralq.py
Follow the prompts to paste in the claim text.

Example Output
yaml
Copy
Edit
Risk Score: 65/100
Risk Level: HIGH

Explanation:
This claim shows several high-risk indicators, including contradictory timelines and lack of supporting evidence...

Recommendations:
- Assign to experienced adjuster for detailed review
- Request additional documentation and evidence
- Conduct phone interview with claimant

Detected Indicators:
- narrative_consistency: Contradictory timeline details
  Evidence: Incident date conflicts within the report
- supporting_evidence: Lack of documentation mentioned
  Evidence: No photos or reports cited in claim
File Structure
insuralq.py: Core AI agent code and CLI interface.

y'


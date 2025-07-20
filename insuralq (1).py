
!pip install openai

import openai
import json

import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

openai.api_key = ""

import openai
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class FraudIndicator:
    category: str
    indicator: str
    weight: float
    detected: bool
    evidence: str

@dataclass
class FraudAnalysis:
    risk_score: float
    risk_level: RiskLevel
    indicators_detected: List[FraudIndicator]
    explanation: str
    recommendations: List[str]

class InsuranceFraudDetector:
    def __init__(self, openai_api_key: str = None):
        if openai_api_key is None:
            import os
            openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")

        if not (openai_api_key.startswith('sk-') or openai_api_key.startswith('sk-proj-')):
            raise ValueError("Invalid API key format. OpenAI keys should start with 'sk-' or 'sk-proj-'")

        try:
            self.client = openai.OpenAI(api_key=openai_api_key)
            self._test_api_key()
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

        self.fraud_indicators = self._initialize_fraud_indicators()

    def _test_api_key(self):
        """Test if the API key is valid with a minimal request"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
        except Exception as e:
            raise ValueError(f"API key validation failed: {str(e)}")

    def _initialize_fraud_indicators(self) -> Dict[str, Dict]:
        """Define fraud indicators with weights and detection criteria"""
        return {
            "narrative_consistency": {
                "weight": 0.15,
                "indicators": [
                    "contradictory_timeline",
                    "changing_story_details",
                    "inconsistent_facts"
                ]
            },
            "detail_level": {
                "weight": 0.12,
                "indicators": [
                    "overly_vague_description",
                    "excessive_unnecessary_detail",
                    "missing_critical_details"
                ]
            },
            "supporting_evidence": {
                "weight": 0.18,
                "indicators": [
                    "no_documentation_mentioned",
                    "refuses_evidence_requests",
                    "convenient_evidence_loss"
                ]
            },
            "timing_patterns": {
                "weight": 0.20,
                "indicators": [
                    "recent_policy_purchase",
                    "claim_after_premium_increase",
                    "suspicious_timing"
                ]
            },
            "language_patterns": {
                "weight": 0.10,
                "indicators": [
                    "over_explaining",
                    "defensive_language",
                    "rehearsed_responses"
                ]
            },
            "claim_characteristics": {
                "weight": 0.15,
                "indicators": [
                    "round_number_amounts",
                    "maximum_coverage_claims",
                    "multiple_recent_claims"
                ]
            },
            "behavioral_flags": {
                "weight": 0.10,
                "indicators": [
                    "pressure_for_quick_settlement",
                    "knowledge_of_policy_details",
                    "evasive_responses"
                ]
            }
        }

    def extract_claim_features(self, claim_text: str) -> Dict[str, Any]:
        """Extract structured features from claim text using GPT-4"""

        feature_extraction_prompt = f"""
        Analyze the following insurance claim text and extract key features. Return your analysis in JSON format:

        Claim Text: {claim_text}

        Extract the following information:
        1. Timeline details (dates, sequence of events)
        2. Damage/loss amounts mentioned
        3. Evidence referenced (photos, reports, witnesses)
        4. Level of detail (specific vs vague descriptions)
        5. Emotional tone and language patterns
        6. Policy-related mentions
        7. Any inconsistencies or contradictions
        8. Urgency indicators

        Format your response as JSON with these keys:
        - timeline
        - amounts
        - evidence
        - detail_level
        - language_tone
        - policy_mentions
        - inconsistencies
        - urgency_indicators
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert insurance claim analyzer. Extract structured information from claim texts."},
                    {"role": "user", "content": feature_extraction_prompt}
                ],
                temperature=0.1
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {}

    def detect_red_flags(self, claim_text: str, features: Dict[str, Any]) -> List[FraudIndicator]:
        """Detect fraud indicators using GPT-4 analysis"""

        red_flag_prompt = f"""
        You are an insurance fraud detection expert. Analyze this claim for fraud indicators:

        Claim Text: {claim_text}

        Extracted Features: {json.dumps(features, indent=2)}

        Check for these specific fraud indicators and provide evidence:

        NARRATIVE CONSISTENCY:
        - Are there contradictory timeline elements?
        - Do story details change or conflict?
        - Are facts inconsistent?

        DETAIL LEVEL:
        - Is the description overly vague?
        - Are there excessive unnecessary details?
        - Are critical details missing?

        SUPPORTING EVIDENCE:
        - Is documentation mentioned or absent?
        - Are there signs of evidence avoidance?
        - Convenient losses of evidence?

        TIMING PATTERNS:
        - Recent policy changes?
        - Suspicious timing of incident?
        - Pattern concerns?

        LANGUAGE PATTERNS:
        - Over-explaining behavior?
        - Defensive language?
        - Rehearsed-sounding responses?

        CLAIM CHARACTERISTICS:
        - Round number amounts?
        - Maximum coverage claims?
        - Multiple recent claims mentioned?

        BEHAVIORAL FLAGS:
        - Pressure for quick settlement?
        - Unusual policy knowledge?
        - Evasive responses?

        For each category, respond with:
        - detected: true/false
        - confidence: 0.0-1.0
        - evidence: specific text or pattern that supports detection
        - explanation: why this indicates potential fraud

        Format as JSON with categories as keys.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert fraud detection analyst with 20 years of experience."},
                    {"role": "user", "content": red_flag_prompt}
                ],
                temperature=0.1
            )

            analysis = json.loads(response.choices[0].message.content)
            indicators = []

            for category, data in analysis.items():
                if data.get('detected', False):
                    indicator = FraudIndicator(
                        category=category,
                        indicator=data.get('explanation', ''),
                        weight=self.fraud_indicators.get(category, {}).get('weight', 0.1),
                        detected=True,
                        evidence=data.get('evidence', '')
                    )
                    indicators.append(indicator)

            return indicators

        except Exception as e:
            print(f"Error detecting red flags: {e}")
            return []

    def calculate_risk_score(self, indicators: List[FraudIndicator]) -> Tuple[float, RiskLevel]:
        """Calculate fraud risk score based on detected indicators"""

        if not indicators:
            return 0.0, RiskLevel.LOW

        total_score = sum(indicator.weight for indicator in indicators if indicator.detected)

        risk_score = min(total_score * 100, 100)

        if risk_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 30:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return risk_score, risk_level

    def generate_explanation(self, indicators: List[FraudIndicator], risk_score: float) -> str:

        if not indicators:
            return "No significant fraud indicators detected. Claim appears legitimate based on text analysis."

        explanation_prompt = f"""
        Generate a clear, professional explanation for an insurance adjuster about why this claim has been flagged for potential fraud.

        Risk Score: {risk_score}/100

        Detected Indicators:
        {json.dumps([{
            'category': ind.category,
            'indicator': ind.indicator,
            'evidence': ind.evidence
        } for ind in indicators], indent=2)}

        Provide:
        1. A summary of the overall risk assessment
        2. Key concerns identified
        3. Specific evidence supporting each concern
        4. Professional tone suitable for insurance professionals

        Keep explanation concise but comprehensive.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an insurance fraud analyst writing professional assessments."},
                    {"role": "user", "content": explanation_prompt}
                ],
                temperature=0.2
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Risk assessment generated based on {len(indicators)} fraud indicators detected."

    def generate_recommendations(self, risk_level: RiskLevel, indicators: List[FraudIndicator]) -> List[str]:

        recommendations = []

        try:
            if risk_level == RiskLevel.CRITICAL:
                recommendations.extend([
                    "Immediately escalate to Special Investigation Unit (SIU)",
                    "Conduct thorough field investigation",
                    "Request comprehensive documentation",
                    "Consider surveillance if appropriate",
                    "Delay settlement pending investigation"
                ])
            elif risk_level == RiskLevel.HIGH:
                recommendations.extend([
                    "Assign to experienced adjuster for detailed review",
                    "Request additional documentation and evidence",
                    "Conduct phone interview with claimant",
                    "Verify all facts and timeline independently",
                    "Consider desktop investigation"
                ])
            elif risk_level == RiskLevel.MEDIUM:
                recommendations.extend([
                    "Enhanced documentation review required",
                    "Follow up with clarifying questions",
                    "Verify key facts through independent sources",
                    "Standard investigation with extra attention to flagged areas"
                ])
            else:
                recommendations.extend([
                    "Process with standard investigation procedures",
                    "Document file notes regarding analysis",
                    "Proceed with normal claim handling"
                ])

            for indicator in indicators:
                if "evidence" in indicator.category.lower():
                    recommendations.append("Request and verify all supporting documentation")
                if "timing" in indicator.category.lower():
                    recommendations.append("Investigate timeline and policy history")
                if "narrative" in indicator.category.lower():
                    recommendations.append("Conduct detailed statement review for consistency")

            return list(set(recommendations))

        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Manual review required due to recommendation generation error"]

    def analyze_claim(self, claim_text: str) -> FraudAnalysis:

        try:
            print("Extracting features...")
            features = self.extract_claim_features(claim_text)

            print("Detecting red flags...")
            indicators = self.detect_red_flags(claim_text, features)

            print(f"Found {len(indicators)} indicators")
            risk_score, risk_level = self.calculate_risk_score(indicators)

            print(f"Risk score: {risk_score}, Risk level: {risk_level}")

            explanation = self.generate_explanation(indicators, risk_score)

            print("Generating recommendations...")
            recommendations = self.generate_recommendations(risk_level, indicators)

            return FraudAnalysis(
                risk_score=risk_score,
                risk_level=risk_level,
                indicators_detected=indicators,
                explanation=explanation,
                recommendations=recommendations
            )

        except Exception as e:
            print(f"Error in analyze_claim: {str(e)}")
            return FraudAnalysis(
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
                indicators_detected=[],
                explanation=f"Error during analysis: {str(e)}",
                recommendations=["Manual review required due to analysis error"]
            )

def main():
    import os

    api_key = ""

    try:
        detector = InsuranceFraudDetector(openai_api_key=api_key)


        print("Please enter the insurance claim text (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
                lines.append(line)
            else:
                lines.append(line)

        claim_text = "\n".join(lines).strip()

        if not claim_text:
            raise ValueError("Claim text cannot be empty. Please provide a valid claim description.")

        analysis = detector.analyze_claim(claim_text)

        print(f"Risk Score: {analysis.risk_score}/100")
        print(f"Risk Level: {analysis.risk_level.value}")
        print(f"\nExplanation:\n{analysis.explanation}")
        print(f"\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"- {rec}")

        print(f"\nDetected Indicators:")
        for indicator in analysis.indicators_detected:
            print(f"- {indicator.category}: {indicator.indicator}")
            print(f"  Evidence: {indicator.evidence}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
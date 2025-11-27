"""
NFL Model Audit & Verification System
======================================
Creates transparent, verifiable logs that a non-technical user can understand.

Design Principles:
1. Every API call is logged with timestamp and response sample
2. Every calculation shows inputs -> formula -> output
3. Warnings are human-readable, not error codes
4. Everything exportable as PDF/HTML report
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd


class VerificationStatus(Enum):
    """Traffic light status for quick scanning."""
    VERIFIED = "Verified"
    WARNING = "Warning"
    FAILED = "Failed"
    PENDING = "Pending"
    SKIPPED = "Skipped"


@dataclass
class APICallLog:
    """Record of a single API call - proves we actually fetched data."""
    timestamp: str
    endpoint: str
    team: Optional[str]
    status_code: int
    response_hash: str  # SHA256 of response for integrity
    response_sample: str  # First 500 chars for human verification
    latency_ms: int
    cached: bool

    def to_human_readable(self) -> str:
        """Format for non-technical user."""
        cached_note = " (from cache)" if self.cached else " (fresh call)"
        return f"""
API Call at {self.timestamp}{cached_note}
- Endpoint: {self.endpoint}
- Team: {self.team or 'All teams'}
- Response: {self.status_code} {'OK' if self.status_code == 200 else 'ERROR'}
- Speed: {self.latency_ms}ms
- Data fingerprint: {self.response_hash[:12]}...
- Sample: "{self.response_sample[:200]}..."
"""


@dataclass
class CalculationStep:
    """Single step in a calculation - shows the math."""
    step_number: int
    description: str
    inputs: Dict[str, Any]
    formula: str
    output: Any

    def to_human_readable(self) -> str:
        inputs_str = ", ".join(f"{k}={v}" for k, v in self.inputs.items())
        return f"""
Step {self.step_number}: {self.description}
- Inputs: {inputs_str}
- Formula: {self.formula}
- Result: {self.output}
"""


@dataclass
class CalculationAudit:
    """Full audit trail for one team's score calculation."""
    team: str
    final_score: float
    profile: str
    timestamp: str
    steps: List[CalculationStep] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)

    def to_human_readable(self) -> str:
        steps_text = "\n".join(s.to_human_readable() for s in self.steps)
        warnings_text = "\n".join(f"- {w}" for w in self.warnings) if self.warnings else "None"
        sources_text = "\n".join(f"- {s}" for s in self.data_sources)

        return f"""
# Calculation Audit: {self.team}
Final Score: {self.final_score:.1f} (Profile: {self.profile})
Generated: {self.timestamp}

## Data Sources Used
{sources_text}

## Warnings
{warnings_text}

## Calculation Steps
{steps_text}
"""


@dataclass
class DataFreshnessCheck:
    """Proves data is recent, not stale."""
    source: str
    last_fetched: str
    age_minutes: int
    max_age_minutes: int
    status: VerificationStatus

    def to_human_readable(self) -> str:
        if self.age_minutes < 60:
            age_str = f"{self.age_minutes} minutes ago"
        elif self.age_minutes < 1440:
            age_str = f"{self.age_minutes // 60} hours ago"
        else:
            age_str = f"{self.age_minutes // 1440} days ago"

        return f"""
{self.source}: {self.status.value}
- Last updated: {self.last_fetched} ({age_str})
- Max allowed age: {self.max_age_minutes} minutes
"""


@dataclass
class CrossValidationCheck:
    """Compares two sources to verify they agree."""
    check_name: str
    source_a: str
    source_a_value: Any
    source_b: str
    source_b_value: Any
    match: bool
    tolerance: Optional[float]
    explanation: str
    status: VerificationStatus

    def to_human_readable(self) -> str:
        match_str = "MATCH" if self.match else "MISMATCH"
        return f"""
{self.check_name}: {match_str}
- {self.source_a}: {self.source_a_value}
- {self.source_b}: {self.source_b_value}
- {self.explanation}
"""


@dataclass
class SanityCheck:
    """Verifies outputs are in expected ranges."""
    check_name: str
    value: Any
    expected_min: Optional[float]
    expected_max: Optional[float]
    passed: bool
    explanation: str
    status: VerificationStatus

    def to_human_readable(self) -> str:
        range_str = ""
        if self.expected_min is not None and self.expected_max is not None:
            range_str = f"Expected range: [{self.expected_min}, {self.expected_max}]"
        elif self.expected_min is not None:
            range_str = f"Expected minimum: {self.expected_min}"
        elif self.expected_max is not None:
            range_str = f"Expected maximum: {self.expected_max}"

        return f"""
{self.check_name}: {self.status.value}
- Value: {self.value}
- {range_str}
- {self.explanation}
"""


class AuditLogger:
    """
    Central audit logger that collects all verification data.

    Usage:
        logger = AuditLogger()
        logger.log_api_call(...)
        logger.log_calculation(...)
        report = logger.generate_report()
    """

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()

        self.api_calls: List[APICallLog] = []
        self.calculations: List[CalculationAudit] = []
        self.freshness_checks: List[DataFreshnessCheck] = []
        self.cross_validations: List[CrossValidationCheck] = []
        self.sanity_checks: List[SanityCheck] = []
        self.warnings: List[Dict] = []
        self.errors: List[Dict] = []

    def log_api_call(self, endpoint: str, team: Optional[str],
                     status_code: int, response_body: str,
                     latency_ms: int, cached: bool = False):
        """Log an API call with proof of fetch."""

        # Create hash for integrity verification
        response_hash = hashlib.sha256(response_body.encode()).hexdigest()

        # Sample first 500 chars for human review
        response_sample = response_body[:500] if len(response_body) > 500 else response_body

        log = APICallLog(
            timestamp=datetime.now().isoformat(),
            endpoint=endpoint,
            team=team,
            status_code=status_code,
            response_hash=response_hash,
            response_sample=response_sample,
            latency_ms=latency_ms,
            cached=cached
        )
        self.api_calls.append(log)
        return log

    def log_calculation(self, team: str, profile: str) -> CalculationAudit:
        """Start a calculation audit for a team."""
        audit = CalculationAudit(
            team=team,
            final_score=0.0,
            profile=profile,
            timestamp=datetime.now().isoformat(),
            steps=[],
            warnings=[],
            data_sources=[]
        )
        self.calculations.append(audit)
        return audit

    def add_calculation_step(self, audit: CalculationAudit,
                            description: str, inputs: Dict,
                            formula: str, output: Any):
        """Add a step to a calculation audit."""
        step = CalculationStep(
            step_number=len(audit.steps) + 1,
            description=description,
            inputs=inputs,
            formula=formula,
            output=output
        )
        audit.steps.append(step)
        return step

    def check_freshness(self, source: str, last_fetched: datetime,
                       max_age_minutes: int = 240) -> DataFreshnessCheck:
        """Check if data source is fresh enough."""
        age = datetime.now() - last_fetched
        age_minutes = int(age.total_seconds() / 60)

        if age_minutes <= max_age_minutes:
            status = VerificationStatus.VERIFIED
        elif age_minutes <= max_age_minutes * 2:
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.FAILED

        check = DataFreshnessCheck(
            source=source,
            last_fetched=last_fetched.isoformat(),
            age_minutes=age_minutes,
            max_age_minutes=max_age_minutes,
            status=status
        )
        self.freshness_checks.append(check)
        return check

    def cross_validate(self, check_name: str,
                      source_a: str, value_a: Any,
                      source_b: str, value_b: Any,
                      tolerance: float = 0.0) -> CrossValidationCheck:
        """Compare two sources to verify agreement."""

        # Determine if they match
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            match = abs(value_a - value_b) <= tolerance
            explanation = f"Difference: {abs(value_a - value_b):.2f} (tolerance: {tolerance})"
        else:
            match = str(value_a).lower() == str(value_b).lower()
            explanation = "String comparison (case-insensitive)"

        status = VerificationStatus.VERIFIED if match else VerificationStatus.WARNING

        check = CrossValidationCheck(
            check_name=check_name,
            source_a=source_a,
            source_a_value=value_a,
            source_b=source_b,
            source_b_value=value_b,
            match=match,
            tolerance=tolerance,
            explanation=explanation,
            status=status
        )
        self.cross_validations.append(check)
        return check

    def sanity_check(self, check_name: str, value: Any,
                    expected_min: float = None,
                    expected_max: float = None) -> SanityCheck:
        """Verify a value is in expected range."""

        passed = True
        explanation = "Value is within expected range"

        if expected_min is not None and value < expected_min:
            passed = False
            explanation = f"Value {value} is below minimum {expected_min}"
        elif expected_max is not None and value > expected_max:
            passed = False
            explanation = f"Value {value} is above maximum {expected_max}"

        status = VerificationStatus.VERIFIED if passed else VerificationStatus.WARNING

        check = SanityCheck(
            check_name=check_name,
            value=value,
            expected_min=expected_min,
            expected_max=expected_max,
            passed=passed,
            explanation=explanation,
            status=status
        )
        self.sanity_checks.append(check)
        return check

    def add_warning(self, category: str, message: str, team: str = None):
        """Add a warning that will appear in the report."""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'message': message,
            'team': team
        })

    def add_error(self, category: str, message: str, team: str = None):
        """Add an error that will appear in the report."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'message': message,
            'team': team
        })

    def get_overall_status(self) -> VerificationStatus:
        """Get overall verification status."""
        if self.errors:
            return VerificationStatus.FAILED

        # Check all verification results
        all_checks = (
            [c.status for c in self.freshness_checks] +
            [c.status for c in self.cross_validations] +
            [c.status for c in self.sanity_checks]
        )

        if any(s == VerificationStatus.FAILED for s in all_checks):
            return VerificationStatus.FAILED
        elif any(s == VerificationStatus.WARNING for s in all_checks):
            return VerificationStatus.WARNING
        elif all_checks:
            return VerificationStatus.VERIFIED
        else:
            return VerificationStatus.PENDING

    def generate_summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'overall_status': self.get_overall_status().value,
            'api_calls': len(self.api_calls),
            'api_calls_cached': sum(1 for c in self.api_calls if c.cached),
            'api_calls_fresh': sum(1 for c in self.api_calls if not c.cached),
            'calculations': len(self.calculations),
            'freshness_checks': {
                'total': len(self.freshness_checks),
                'passed': sum(1 for c in self.freshness_checks if c.status == VerificationStatus.VERIFIED),
                'warnings': sum(1 for c in self.freshness_checks if c.status == VerificationStatus.WARNING),
                'failed': sum(1 for c in self.freshness_checks if c.status == VerificationStatus.FAILED),
            },
            'cross_validations': {
                'total': len(self.cross_validations),
                'matched': sum(1 for c in self.cross_validations if c.match),
                'mismatched': sum(1 for c in self.cross_validations if not c.match),
            },
            'sanity_checks': {
                'total': len(self.sanity_checks),
                'passed': sum(1 for c in self.sanity_checks if c.passed),
                'failed': sum(1 for c in self.sanity_checks if not c.passed),
            },
            'warnings': len(self.warnings),
            'errors': len(self.errors),
        }

    def generate_report_markdown(self) -> str:
        """Generate full human-readable report in Markdown."""
        summary = self.generate_summary()

        report = f"""
# NFL Model Verification Report

**Session ID:** {summary['session_id']}
**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
**Overall Status:** {summary['overall_status']}

---

## Executive Summary

| Metric | Count |
|--------|-------|
| API Calls | {summary['api_calls']} ({summary['api_calls_fresh']} fresh, {summary['api_calls_cached']} cached) |
| Teams Calculated | {summary['calculations']} |
| Warnings | {summary['warnings']} |
| Errors | {summary['errors']} |

### Verification Results

| Check Type | Passed | Warnings | Failed |
|------------|--------|----------|--------|
| Data Freshness | {summary['freshness_checks']['passed']} | {summary['freshness_checks']['warnings']} | {summary['freshness_checks']['failed']} |
| Cross-Validation | {summary['cross_validations']['matched']} | - | {summary['cross_validations']['mismatched']} |
| Sanity Checks | {summary['sanity_checks']['passed']} | - | {summary['sanity_checks']['failed']} |

---

## Data Freshness

*These checks verify we're using recent data, not stale cache.*

"""
        for check in self.freshness_checks:
            report += check.to_human_readable() + "\n"

        report += """
---

## Cross-Validation Checks

*These checks compare multiple data sources to verify agreement.*

"""
        for check in self.cross_validations:
            report += check.to_human_readable() + "\n"

        report += """
---

## Sanity Checks

*These checks verify outputs are in expected ranges.*

"""
        for check in self.sanity_checks:
            report += check.to_human_readable() + "\n"

        if self.warnings:
            report += """
---

## Warnings

"""
            for w in self.warnings:
                team_str = f" ({w['team']})" if w['team'] else ""
                report += f"- **{w['category']}**{team_str}: {w['message']}\n"

        if self.errors:
            report += """
---

## Errors

"""
            for e in self.errors:
                team_str = f" ({e['team']})" if e['team'] else ""
                report += f"- **{e['category']}**{team_str}: {e['message']}\n"

        report += """
---

## API Call Log

*Proof that we actually called external data sources.*

"""
        for call in self.api_calls[-10:]:  # Last 10 calls
            report += call.to_human_readable() + "\n"

        if len(self.api_calls) > 10:
            report += f"\n*...and {len(self.api_calls) - 10} more calls (see full log)*\n"

        return report

    def save_full_log(self, filepath: str = 'data/audit_log.json'):
        """Save complete audit log to JSON for detailed analysis."""
        log_data = {
            'summary': self.generate_summary(),
            'api_calls': [asdict(c) for c in self.api_calls],
            'calculations': [asdict(c) for c in self.calculations],
            'freshness_checks': [asdict(c) for c in self.freshness_checks],
            'cross_validations': [asdict(c) for c in self.cross_validations],
            'sanity_checks': [asdict(c) for c in self.sanity_checks],
            'warnings': self.warnings,
            'errors': self.errors,
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        return filepath


# Global logger instance
_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def reset_audit_logger():
    """Reset the audit logger for a new session."""
    global _audit_logger
    _audit_logger = AuditLogger()
    return _audit_logger


# Pre-defined sanity check ranges based on NFL data
SANITY_RANGES = {
    'qb_epa_per_dropback': {'min': -0.5, 'max': 0.5},
    'cpoe': {'min': -10, 'max': 15},
    'team_score': {'min': 0, 'max': 100},
    'win_probability': {'min': 0, 'max': 1},
    'pressure_rate': {'min': 10, 'max': 50},
    '3rd_down_rate': {'min': 20, 'max': 60},
    'red_zone_rate': {'min': 30, 'max': 80},
    'games_played': {'min': 1, 'max': 17},
    'pass_attempts': {'min': 100, 'max': 700},
}


def run_standard_sanity_checks(logger: AuditLogger, team: str, data: Dict):
    """Run standard sanity checks on team data."""

    for field, ranges in SANITY_RANGES.items():
        if field in data and data[field] is not None:
            logger.sanity_check(
                check_name=f"{team} - {field}",
                value=data[field],
                expected_min=ranges.get('min'),
                expected_max=ranges.get('max')
            )


def audited_api_call(logger: AuditLogger, url: str, team: str = None):
    """Wrapper that logs API calls for audit trail."""
    import requests
    import time

    start_time = time.time()

    try:
        response = requests.get(url, timeout=10)
        latency_ms = int((time.time() - start_time) * 1000)

        logger.log_api_call(
            endpoint=url,
            team=team,
            status_code=response.status_code,
            response_body=response.text,
            latency_ms=latency_ms,
            cached=False
        )

        return response

    except Exception as e:
        logger.add_error(
            category="API_CALL",
            message=f"Failed to call {url}: {str(e)}",
            team=team
        )
        raise


if __name__ == "__main__":
    # Example usage
    logger = AuditLogger()

    # Simulate some checks
    logger.sanity_check("Test Score", 75, expected_min=0, expected_max=100)
    logger.add_warning("DATA", "Using cached data from 2 hours ago")

    print(logger.generate_report_markdown())

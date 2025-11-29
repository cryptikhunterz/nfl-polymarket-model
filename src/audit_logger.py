#!/usr/bin/env python3
"""
Audit Logger for Factor Calculations

Tracks all factor calculations with detailed logging:
- Data sources used
- Missing data fields
- Calculation errors
- Data quality metrics
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

AUDIT_DIR = Path(__file__).parent.parent / "data" / "audit"


class DataSource(Enum):
    """Data source types for factor calculations."""
    FTN_EPA = "FTN EPA"
    ESPN_2025 = "ESPN 2025 Stats"
    NFL_DATA_PY = "nfl_data_py historical"
    ESPN_ROSTER = "ESPN Roster API"
    PFF_GRADES = "PFF Grades"
    ELOS = "Elo Ratings"
    PBP = "Play-by-Play"
    BASELINE = "Baseline/Default"


class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MISSING = "missing"


@dataclass
class FactorAuditEntry:
    """Single audit entry for a factor calculation."""
    factor_name: str
    team: str
    timestamp: str
    success: bool
    score: Optional[float] = None
    data_sources: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    data_quality: str = "unknown"
    sample_size: int = 0
    raw_values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FactorAuditReport:
    """Complete audit report for all factor calculations."""
    run_id: str
    timestamp: str
    season: int
    week: Optional[int]
    factors_calculated: int = 0
    factors_succeeded: int = 0
    factors_failed: int = 0
    teams_processed: int = 0
    data_sources_used: List[str] = field(default_factory=list)
    missing_data_summary: Dict[str, int] = field(default_factory=dict)
    entries: List[FactorAuditEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'season': self.season,
            'week': self.week,
            'factors_calculated': self.factors_calculated,
            'factors_succeeded': self.factors_succeeded,
            'factors_failed': self.factors_failed,
            'teams_processed': self.teams_processed,
            'data_sources_used': self.data_sources_used,
            'missing_data_summary': self.missing_data_summary,
            'entries': [e.to_dict() for e in self.entries],
            'errors': self.errors
        }


class AuditLogger:
    """
    Logger for factor calculation auditing.

    Usage:
        audit = AuditLogger(season=2024, week=12)
        audit.log_factor_start("qb_quality", "KC")
        audit.log_data_source("FTN EPA")
        audit.log_missing_field("cpoe")
        audit.log_factor_complete("qb_quality", "KC", score=85.3, success=True)
        audit.save_report()
    """

    def __init__(self, season: int = 2024, week: int = None):
        self.season = season
        self.week = week
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = datetime.now().isoformat()

        # Current factor being logged
        self._current_factor: Optional[str] = None
        self._current_team: Optional[str] = None
        self._current_entry: Optional[FactorAuditEntry] = None

        # All entries
        self.entries: List[FactorAuditEntry] = []
        self.data_sources_used: set = set()
        self.missing_data_counts: Dict[str, int] = {}
        self.errors: List[str] = []

        # Teams processed
        self.teams_processed: set = set()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup file and console logging."""
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"audit_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = AUDIT_DIR / f"factor_audit_{self.run_id}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler (less verbose)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Audit started: run_id={self.run_id}, season={self.season}, week={self.week}")

    def log_factor_start(self, factor_name: str, team: str):
        """Start logging a new factor calculation."""
        self._current_factor = factor_name
        self._current_team = team
        self._current_entry = FactorAuditEntry(
            factor_name=factor_name,
            team=team,
            timestamp=datetime.now().isoformat(),
            success=False
        )
        self.teams_processed.add(team)
        self.logger.debug(f"Starting {factor_name} for {team}")

    def log_data_source(self, source: str, quality: str = "high"):
        """Log a data source being used."""
        if self._current_entry:
            self._current_entry.data_sources.append(source)
            self._current_entry.data_quality = quality
        self.data_sources_used.add(source)
        self.logger.debug(f"  Using data source: {source} (quality: {quality})")

    def log_missing_field(self, field_name: str, critical: bool = False):
        """Log a missing data field."""
        if self._current_entry:
            self._current_entry.missing_fields.append(field_name)

        # Track counts
        key = f"{self._current_factor}:{field_name}" if self._current_factor else field_name
        self.missing_data_counts[key] = self.missing_data_counts.get(key, 0) + 1

        level = logging.WARNING if critical else logging.DEBUG
        self.logger.log(level, f"  Missing field: {field_name}")

    def log_warning(self, message: str):
        """Log a warning for the current factor."""
        if self._current_entry:
            self._current_entry.warnings.append(message)
        self.logger.warning(f"  {message}")

    def log_raw_value(self, key: str, value: Any):
        """Log a raw calculated value."""
        if self._current_entry:
            self._current_entry.raw_values[key] = value
        self.logger.debug(f"  {key}: {value}")

    def log_sample_size(self, n: int):
        """Log the sample size used for calculation."""
        if self._current_entry:
            self._current_entry.sample_size = n
        self.logger.debug(f"  Sample size: {n}")

    def log_factor_complete(self, factor_name: str, team: str,
                           score: float = None, success: bool = True,
                           error: str = None):
        """Complete logging for a factor calculation."""
        if self._current_entry:
            self._current_entry.success = success
            self._current_entry.score = score
            self._current_entry.error = error
            self.entries.append(self._current_entry)

            if success:
                self.logger.info(f"  {team} {factor_name}: {score:.2f}" if score else f"  {team} {factor_name}: OK")
            else:
                self.logger.error(f"  {team} {factor_name}: FAILED - {error}")
                self.errors.append(f"{team}/{factor_name}: {error}")

        # Reset current
        self._current_entry = None
        self._current_factor = None
        self._current_team = None

    def log_error(self, message: str, exception: Exception = None):
        """Log an error."""
        error_msg = f"{message}: {str(exception)}" if exception else message
        self.errors.append(error_msg)
        self.logger.error(error_msg)

    def get_report(self) -> FactorAuditReport:
        """Generate the audit report."""
        succeeded = sum(1 for e in self.entries if e.success)
        failed = sum(1 for e in self.entries if not e.success)

        return FactorAuditReport(
            run_id=self.run_id,
            timestamp=self.timestamp,
            season=self.season,
            week=self.week,
            factors_calculated=len(self.entries),
            factors_succeeded=succeeded,
            factors_failed=failed,
            teams_processed=len(self.teams_processed),
            data_sources_used=list(self.data_sources_used),
            missing_data_summary=self.missing_data_counts,
            entries=self.entries,
            errors=self.errors
        )

    def save_report(self, filename: str = None) -> Path:
        """Save the audit report to JSON."""
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

        report = self.get_report()

        if filename is None:
            filename = f"factor_audit_{self.run_id}.json"

        filepath = AUDIT_DIR / filename

        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Audit report saved: {filepath}")
        return filepath

    def print_summary(self):
        """Print a summary of the audit."""
        report = self.get_report()

        print("\n" + "=" * 70)
        print("FACTOR CALCULATION AUDIT SUMMARY")
        print("=" * 70)
        print(f"Run ID: {report.run_id}")
        print(f"Season: {report.season}, Week: {report.week}")
        print(f"Timestamp: {report.timestamp}")
        print("-" * 70)
        print(f"Teams Processed: {report.teams_processed}")
        print(f"Factors Calculated: {report.factors_calculated}")
        print(f"  - Succeeded: {report.factors_succeeded}")
        print(f"  - Failed: {report.factors_failed}")
        print("-" * 70)
        print("Data Sources Used:")
        for src in sorted(report.data_sources_used):
            print(f"  - {src}")
        print("-" * 70)

        if report.missing_data_summary:
            print("Missing Data Summary:")
            for field, count in sorted(report.missing_data_summary.items(),
                                       key=lambda x: -x[1])[:10]:
                print(f"  - {field}: {count} occurrences")

        if report.errors:
            print("-" * 70)
            print(f"Errors ({len(report.errors)}):")
            for err in report.errors[:5]:
                print(f"  - {err}")
            if len(report.errors) > 5:
                print(f"  ... and {len(report.errors) - 5} more")

        print("=" * 70)

    def get_factor_summary(self) -> Dict[str, Dict]:
        """Get summary statistics by factor type."""
        summary = {}

        for entry in self.entries:
            factor = entry.factor_name
            if factor not in summary:
                summary[factor] = {
                    'count': 0,
                    'succeeded': 0,
                    'failed': 0,
                    'avg_score': 0,
                    'scores': [],
                    'missing_fields': set(),
                    'data_sources': set()
                }

            summary[factor]['count'] += 1
            if entry.success:
                summary[factor]['succeeded'] += 1
                if entry.score is not None:
                    summary[factor]['scores'].append(entry.score)
            else:
                summary[factor]['failed'] += 1

            summary[factor]['missing_fields'].update(entry.missing_fields)
            summary[factor]['data_sources'].update(entry.data_sources)

        # Calculate averages
        for factor in summary:
            scores = summary[factor]['scores']
            if scores:
                summary[factor]['avg_score'] = sum(scores) / len(scores)
            summary[factor]['missing_fields'] = list(summary[factor]['missing_fields'])
            summary[factor]['data_sources'] = list(summary[factor]['data_sources'])
            del summary[factor]['scores']

        return summary


# Convenience function
def create_audit_logger(season: int = 2024, week: int = None) -> AuditLogger:
    """Create a new audit logger instance."""
    return AuditLogger(season=season, week=week)


if __name__ == "__main__":
    # Test the audit logger
    audit = AuditLogger(season=2024, week=12)

    # Simulate some factor calculations
    for team in ["KC", "BUF", "DET"]:
        audit.log_factor_start("qb_quality", team)
        audit.log_data_source("FTN EPA", quality="high")
        audit.log_raw_value("epa_dropback", 0.15)
        audit.log_sample_size(11)
        audit.log_factor_complete("qb_quality", team, score=85.0, success=True)

    # Simulate a failure
    audit.log_factor_start("qb_quality", "CLE")
    audit.log_data_source("ESPN 2025 Stats", quality="medium")
    audit.log_missing_field("cpoe", critical=True)
    audit.log_warning("Small sample size")
    audit.log_factor_complete("qb_quality", "CLE", score=45.0, success=True)

    audit.print_summary()
    audit.save_report()

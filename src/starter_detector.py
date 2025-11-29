"""
NFL Starter Detector

Detects current starting QBs for each team using:
1. ESPN depth chart API
2. ESPN injury reports
3. Play-by-play snap count validation

Cross-validates multiple sources for confidence scoring.
"""

import logging
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False

try:
    from team_config import NFL_TEAMS, ESPN_TEAM_IDS, normalize_team_name
except ImportError:
    from src.team_config import NFL_TEAMS, ESPN_TEAM_IDS, normalize_team_name

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence level for starter detection."""
    HIGH = "HIGH"       # Multiple sources agree, no injury concerns
    MEDIUM = "MEDIUM"   # Sources agree but injury concerns OR single source
    LOW = "LOW"         # Sources disagree or limited data


class InjuryStatus(Enum):
    """NFL injury designations."""
    HEALTHY = "Healthy"
    QUESTIONABLE = "Questionable"
    DOUBTFUL = "Doubtful"
    OUT = "Out"
    IR = "IR"
    PUP = "PUP"
    UNKNOWN = "Unknown"


@dataclass
class StarterInfo:
    """Information about a team's starting QB."""
    team: str                           # Canonical team abbreviation
    player_name: str                    # QB's full name
    player_id: Optional[str] = None     # ESPN player ID if available
    injury_status: InjuryStatus = InjuryStatus.HEALTHY
    injury_detail: Optional[str] = None # e.g., "Ankle"
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    depth_chart_rank: int = 1           # 1 = starter, 2 = backup, etc.
    recent_snap_pct: Optional[float] = None  # From PBP data
    last_game_snaps: Optional[int] = None
    sources: List[str] = field(default_factory=list)  # Which sources confirmed
    last_updated: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None         # Any additional context


# =============================================================================
# ESPN API FETCHER
# =============================================================================

class ESPNFetcher:
    """Fetches data from ESPN APIs."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; NFLPolymarketModel/1.0)"
        })

    def get_depth_chart(self, team: str) -> Optional[Dict]:
        """
        Fetch depth chart for a team.

        Args:
            team: Canonical team abbreviation (e.g., "KC")

        Returns:
            Dict with depth chart data or None on error
        """
        team = normalize_team_name(team)
        if team is None or team not in ESPN_TEAM_IDS:
            logger.error(f"Unknown team: {team}")
            return None

        team_id = ESPN_TEAM_IDS[team]
        url = f"{self.BASE_URL}/teams/{team_id}/depthcharts"

        try:
            logger.debug(f"Fetching depth chart for {team} (ESPN ID: {team_id})")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch depth chart for {team}: {e}")
            return None

    def get_injuries(self, team: str) -> Optional[List[Dict]]:
        """
        Fetch injury report for a team.

        Args:
            team: Canonical team abbreviation

        Returns:
            List of injury records or None on error
        """
        team = normalize_team_name(team)
        if team is None or team not in ESPN_TEAM_IDS:
            logger.error(f"Unknown team: {team}")
            return None

        team_id = ESPN_TEAM_IDS[team]
        url = f"{self.BASE_URL}/teams/{team_id}/injuries"

        try:
            logger.debug(f"Fetching injuries for {team}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract injury list from response
            injuries = []
            if "team" in data and "injuries" in data["team"]:
                injuries = data["team"]["injuries"]
            elif "injuries" in data:
                injuries = data["injuries"]

            return injuries
        except requests.RequestException as e:
            logger.error(f"Failed to fetch injuries for {team}: {e}")
            return None

    def get_roster(self, team: str) -> Optional[List[Dict]]:
        """
        Fetch team roster.

        Args:
            team: Canonical team abbreviation

        Returns:
            List of player records or None on error
        """
        team = normalize_team_name(team)
        if team is None or team not in ESPN_TEAM_IDS:
            logger.error(f"Unknown team: {team}")
            return None

        team_id = ESPN_TEAM_IDS[team]
        url = f"{self.BASE_URL}/teams/{team_id}/roster"

        try:
            logger.debug(f"Fetching roster for {team}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract athletes from roster groups
            athletes = []
            if "athletes" in data:
                for group in data["athletes"]:
                    if "items" in group:
                        athletes.extend(group["items"])

            return athletes
        except requests.RequestException as e:
            logger.error(f"Failed to fetch roster for {team}: {e}")
            return None

    def parse_qb_from_depth_chart(self, depth_data: Dict) -> List[Dict]:
        """
        Extract QB information from depth chart data.

        ESPN depth chart structure:
        - depthchart: array of formations (offense, defense, special teams)
        - Each formation has positions dict keyed by position abbreviation ("qb", "rb", etc.)
        - Each position has athletes array in depth chart order (first = starter)

        Returns list of QBs with their depth chart position.
        """
        qbs = []

        if not depth_data:
            return qbs

        try:
            # Navigate ESPN's depth chart structure - key is 'depthchart' not 'items'
            formations = depth_data.get("depthchart", [])

            for formation in formations:
                positions = formation.get("positions", {})

                # Look for QB position (lowercase 'qb')
                if "qb" not in positions:
                    continue

                qb_data = positions["qb"]
                athletes = qb_data.get("athletes", [])

                for rank, athlete in enumerate(athletes, start=1):
                    qb_info = {
                        "name": athlete.get("displayName", athlete.get("fullName", "Unknown")),
                        "id": str(athlete.get("id", "")),
                        "rank": rank,
                    }
                    qbs.append(qb_info)

                if qbs:
                    break  # Found QBs, no need to check other formations

        except (KeyError, TypeError) as e:
            logger.warning(f"Error parsing depth chart: {e}")

        return qbs

    def parse_qb_injuries(self, injuries: List[Dict]) -> Dict[str, Dict]:
        """
        Extract QB injuries from injury report.

        Returns dict mapping player name to injury info.
        """
        qb_injuries = {}

        if not injuries:
            return qb_injuries

        for injury in injuries:
            try:
                athlete = injury.get("athlete", {})
                position = athlete.get("position", {})
                pos_abbr = position.get("abbreviation", "")

                if pos_abbr.upper() == "QB":
                    name = athlete.get("displayName", athlete.get("fullName", "Unknown"))
                    status_text = injury.get("status", "Unknown")

                    # Map status to enum
                    status_map = {
                        "questionable": InjuryStatus.QUESTIONABLE,
                        "doubtful": InjuryStatus.DOUBTFUL,
                        "out": InjuryStatus.OUT,
                        "injured reserve": InjuryStatus.IR,
                        "ir": InjuryStatus.IR,
                        "pup": InjuryStatus.PUP,
                    }
                    status = status_map.get(status_text.lower(), InjuryStatus.UNKNOWN)

                    qb_injuries[name] = {
                        "status": status,
                        "detail": injury.get("type", {}).get("description", None),
                        "id": str(athlete.get("id", "")),
                    }

            except (KeyError, TypeError) as e:
                logger.warning(f"Error parsing injury: {e}")
                continue

        return qb_injuries


# =============================================================================
# PLAY-BY-PLAY FETCHER
# =============================================================================

class PBPFetcher:
    """Fetches and analyzes play-by-play data for snap counts."""

    def __init__(self):
        if not NFL_DATA_AVAILABLE:
            logger.warning("nfl_data_py not available - PBP validation disabled")

    def get_recent_qb_snaps(self, team: str, weeks: int = 3) -> Optional[Dict]:
        """
        Get QB snap counts from recent games.

        Args:
            team: Canonical team abbreviation
            weeks: Number of recent weeks to analyze

        Returns:
            Dict with QB snap data or None
        """
        if not NFL_DATA_AVAILABLE:
            return None

        team = normalize_team_name(team)
        if team is None:
            return None

        try:
            # Get current season
            current_year = datetime.now().year
            # If before September, use previous year
            if datetime.now().month < 9:
                current_year -= 1

            logger.debug(f"Fetching PBP data for {team}, season {current_year}")

            # Load play-by-play data
            pbp = nfl.import_pbp_data([current_year])

            if pbp is None or len(pbp) == 0:
                logger.warning(f"No PBP data available for {current_year}")
                return None

            # Filter to team's offensive plays
            team_plays = pbp[
                (pbp["posteam"] == team) &
                (pbp["play_type"].isin(["pass", "run"]))
            ]

            if len(team_plays) == 0:
                logger.warning(f"No plays found for {team}")
                return None

            # Get recent weeks
            max_week = team_plays["week"].max()
            min_week = max(1, max_week - weeks + 1)
            recent_plays = team_plays[team_plays["week"] >= min_week]

            # Count snaps by passer (for pass plays)
            qb_snaps = {}
            pass_plays = recent_plays[recent_plays["play_type"] == "pass"]

            if "passer_player_name" in pass_plays.columns:
                snap_counts = pass_plays.groupby("passer_player_name").size()
                total_pass_plays = len(pass_plays)

                for qb_name, snaps in snap_counts.items():
                    if qb_name and snaps > 0:
                        qb_snaps[qb_name] = {
                            "snaps": int(snaps),
                            "snap_pct": snaps / total_pass_plays if total_pass_plays > 0 else 0,
                            "weeks_analyzed": int(max_week - min_week + 1),
                        }

            return qb_snaps if qb_snaps else None

        except Exception as e:
            logger.error(f"Error fetching PBP data for {team}: {e}")
            return None


# =============================================================================
# STARTER DETECTOR
# =============================================================================

class StarterDetector:
    """
    Detects starting QBs by cross-referencing multiple sources.

    Priority:
    1. ESPN depth chart (primary source)
    2. ESPN injury report (modifies confidence)
    3. PBP snap counts (validation)
    """

    def __init__(self, use_pbp_validation: bool = True):
        """
        Initialize detector.

        Args:
            use_pbp_validation: Whether to validate with PBP data
        """
        self.espn = ESPNFetcher()
        self.pbp = PBPFetcher() if use_pbp_validation else None
        self._cache: Dict[str, StarterInfo] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_minutes = 30

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        age_minutes = (datetime.now() - self._cache_time).total_seconds() / 60
        return age_minutes < self._cache_ttl_minutes

    def get_starter(self, team: str, force_refresh: bool = False) -> Optional[StarterInfo]:
        """
        Get starting QB for a team.

        Args:
            team: Team abbreviation (will be normalized)
            force_refresh: Bypass cache if True

        Returns:
            StarterInfo or None if unable to determine
        """
        team = normalize_team_name(team)
        if team is None:
            logger.error(f"Invalid team: {team}")
            return None

        # Check cache
        if not force_refresh and self._is_cache_valid() and team in self._cache:
            logger.debug(f"Returning cached starter for {team}")
            return self._cache[team]

        logger.info(f"Detecting starter for {team}")
        sources = []

        # 1. Get depth chart
        depth_data = self.espn.get_depth_chart(team)
        qbs = self.espn.parse_qb_from_depth_chart(depth_data) if depth_data else []

        if not qbs:
            logger.warning(f"No QBs found in depth chart for {team}")
            # Try roster as fallback
            roster = self.espn.get_roster(team)
            if roster:
                for player in roster:
                    pos = player.get("position", {}).get("abbreviation", "")
                    if pos.upper() == "QB":
                        qbs.append({
                            "name": player.get("displayName", "Unknown"),
                            "id": str(player.get("id", "")),
                            "rank": len(qbs) + 1,
                        })
                if qbs:
                    sources.append("roster")
        else:
            sources.append("depth_chart")

        if not qbs:
            logger.error(f"Could not find any QBs for {team}")
            return None

        # Get the #1 QB from depth chart
        starter_qb = qbs[0]
        starter_name = starter_qb["name"]
        starter_id = starter_qb.get("id")
        starter_rank = starter_qb.get("rank", 1)

        # 2. Check injuries
        injuries = self.espn.get_injuries(team)
        qb_injuries = self.espn.parse_qb_injuries(injuries) if injuries else {}

        injury_status = InjuryStatus.HEALTHY
        injury_detail = None

        if starter_name in qb_injuries:
            injury_info = qb_injuries[starter_name]
            injury_status = injury_info["status"]
            injury_detail = injury_info.get("detail")
            sources.append("injury_report")
            logger.info(f"{team} QB {starter_name}: {injury_status.value} ({injury_detail})")

            # If starter is OUT/IR, look for backup
            if injury_status in (InjuryStatus.OUT, InjuryStatus.IR, InjuryStatus.PUP):
                logger.info(f"{team} starter {starter_name} is {injury_status.value}, checking backup")
                for qb in qbs[1:]:
                    backup_name = qb["name"]
                    if backup_name not in qb_injuries:
                        # Backup is healthy
                        starter_name = backup_name
                        starter_id = qb.get("id")
                        starter_rank = qb.get("rank", 2)
                        injury_status = InjuryStatus.HEALTHY
                        injury_detail = f"Filling in for injured {qbs[0]['name']}"
                        logger.info(f"{team} backup {starter_name} expected to start")
                        break
                    else:
                        backup_injury = qb_injuries[backup_name]
                        if backup_injury["status"] not in (InjuryStatus.OUT, InjuryStatus.IR, InjuryStatus.PUP):
                            starter_name = backup_name
                            starter_id = qb.get("id")
                            starter_rank = qb.get("rank", 2)
                            injury_status = backup_injury["status"]
                            injury_detail = backup_injury.get("detail")
                            break

        # 3. Validate with PBP data
        recent_snap_pct = None
        last_game_snaps = None

        if self.pbp:
            pbp_data = self.pbp.get_recent_qb_snaps(team)
            if pbp_data:
                sources.append("pbp_validation")
                # Try to match starter name to PBP names (last name matching)
                starter_last = starter_name.split()[-1].lower() if starter_name else ""
                for pbp_name, data in pbp_data.items():
                    pbp_last = pbp_name.split()[-1].lower() if pbp_name else ""
                    if starter_last and pbp_last and starter_last == pbp_last:
                        recent_snap_pct = data.get("snap_pct")
                        last_game_snaps = data.get("snaps")
                        logger.debug(f"PBP match: {starter_name} -> {pbp_name}, {recent_snap_pct:.1%} snaps")
                        break

        # 4. Determine confidence level
        confidence = self._calculate_confidence(
            sources=sources,
            injury_status=injury_status,
            depth_rank=starter_rank,
            snap_pct=recent_snap_pct,
        )

        # Build StarterInfo
        starter_info = StarterInfo(
            team=team,
            player_name=starter_name,
            player_id=starter_id,
            injury_status=injury_status,
            injury_detail=injury_detail,
            confidence=confidence,
            depth_chart_rank=starter_rank,
            recent_snap_pct=recent_snap_pct,
            last_game_snaps=last_game_snaps,
            sources=sources,
            last_updated=datetime.now(),
        )

        # Cache result
        self._cache[team] = starter_info
        self._cache_time = datetime.now()

        logger.info(
            f"{team}: {starter_name} (Confidence: {confidence.value}, "
            f"Injury: {injury_status.value})"
        )

        return starter_info

    def _calculate_confidence(
        self,
        sources: List[str],
        injury_status: InjuryStatus,
        depth_rank: int,
        snap_pct: Optional[float],
    ) -> ConfidenceLevel:
        """
        Calculate confidence level based on available data.

        HIGH: Multiple sources agree, healthy or minor injury, high snap %
        MEDIUM: Depth chart only, or questionable injury, or lower snap %
        LOW: Sources disagree, doubtful injury, backup starting, or limited data
        """
        score = 0

        # Source agreement
        if "depth_chart" in sources:
            score += 2
        if "pbp_validation" in sources:
            score += 1
        if len(sources) >= 2:
            score += 1

        # Injury impact
        if injury_status == InjuryStatus.HEALTHY:
            score += 2
        elif injury_status == InjuryStatus.QUESTIONABLE:
            score += 0  # Neutral
        elif injury_status == InjuryStatus.DOUBTFUL:
            score -= 2
        elif injury_status in (InjuryStatus.OUT, InjuryStatus.IR):
            score -= 3  # Major concern if we're predicting they start

        # Depth chart position
        if depth_rank == 1:
            score += 1
        elif depth_rank >= 3:
            score -= 2

        # Snap percentage validation
        if snap_pct is not None:
            if snap_pct >= 0.8:
                score += 2
            elif snap_pct >= 0.5:
                score += 1
            elif snap_pct < 0.3:
                score -= 1

        # Convert score to confidence level
        if score >= 5:
            return ConfidenceLevel.HIGH
        elif score >= 2:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def get_all_starters(self, force_refresh: bool = False) -> Dict[str, StarterInfo]:
        """
        Get starting QBs for all 32 teams.

        Args:
            force_refresh: Bypass cache if True

        Returns:
            Dict mapping team abbreviation to StarterInfo
        """
        logger.info("Fetching starters for all 32 NFL teams...")
        starters = {}

        for team in sorted(NFL_TEAMS):
            try:
                info = self.get_starter(team, force_refresh=force_refresh)
                if info:
                    starters[team] = info
                else:
                    logger.warning(f"Could not determine starter for {team}")
            except Exception as e:
                logger.error(f"Error getting starter for {team}: {e}")

        logger.info(f"Found starters for {len(starters)}/32 teams")
        return starters

    def export_to_csv(self, filepath: str, starters: Optional[Dict[str, StarterInfo]] = None):
        """
        Export starter data to CSV.

        Args:
            filepath: Output file path
            starters: Dict of starters (fetches all if not provided)
        """
        import csv

        if starters is None:
            starters = self.get_all_starters()

        headers = [
            "team", "player_name", "player_id", "injury_status",
            "injury_detail", "confidence", "depth_chart_rank",
            "recent_snap_pct", "last_game_snaps", "sources", "last_updated"
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for team in sorted(starters.keys()):
                info = starters[team]
                writer.writerow([
                    info.team,
                    info.player_name,
                    info.player_id or "",
                    info.injury_status.value,
                    info.injury_detail or "",
                    info.confidence.value,
                    info.depth_chart_rank,
                    f"{info.recent_snap_pct:.2f}" if info.recent_snap_pct else "",
                    info.last_game_snaps or "",
                    ",".join(info.sources),
                    info.last_updated.isoformat(),
                ])

        logger.info(f"Exported {len(starters)} starters to {filepath}")

    def print_report(self, starters: Optional[Dict[str, StarterInfo]] = None):
        """Print a formatted report of all starters."""
        if starters is None:
            starters = self.get_all_starters()

        print("\n" + "=" * 70)
        print("NFL STARTING QB REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)

        # Group by confidence
        by_confidence = {
            ConfidenceLevel.HIGH: [],
            ConfidenceLevel.MEDIUM: [],
            ConfidenceLevel.LOW: [],
        }

        for team, info in starters.items():
            by_confidence[info.confidence].append((team, info))

        for confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]:
            teams = by_confidence[confidence]
            if not teams:
                continue

            print(f"\n{confidence.value} CONFIDENCE ({len(teams)} teams):")
            print("-" * 70)

            for team, info in sorted(teams):
                injury_str = ""
                if info.injury_status != InjuryStatus.HEALTHY:
                    injury_str = f" [{info.injury_status.value}"
                    if info.injury_detail:
                        injury_str += f" - {info.injury_detail}"
                    injury_str += "]"

                snap_str = ""
                if info.recent_snap_pct:
                    snap_str = f" ({info.recent_snap_pct:.0%} snaps)"

                print(f"  {team:3s}: {info.player_name:<25}{injury_str}{snap_str}")

        # Summary
        healthy = sum(1 for i in starters.values() if i.injury_status == InjuryStatus.HEALTHY)
        questionable = sum(1 for i in starters.values() if i.injury_status == InjuryStatus.QUESTIONABLE)

        print("\n" + "-" * 70)
        print(f"SUMMARY: {len(starters)} teams, {healthy} healthy, {questionable} questionable")
        print("=" * 70)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def detect_starters():
    """Main function to run starter detection."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    detector = StarterDetector(use_pbp_validation=True)
    starters = detector.get_all_starters()
    detector.print_report(starters)

    # Export to outputs directory
    from pathlib import Path
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    detector.export_to_csv(str(outputs_dir / "starters.csv"), starters)

    return starters


if __name__ == "__main__":
    detect_starters()

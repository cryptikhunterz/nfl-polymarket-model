"""
NFL POLYMARKET MODEL - STREAMLIT UI
===================================
Team profiles and model comparison dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import base64
from io import StringIO, BytesIO

# Try to import Anthropic and PIL
try:
    import anthropic
    from PIL import Image
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    CLAUDE_VISION_AVAILABLE = ANTHROPIC_API_KEY is not None
except ImportError:
    CLAUDE_VISION_AVAILABLE = False
    ANTHROPIC_API_KEY = None

# Source-specific extraction prompts
EXTRACTION_PROMPTS = {
    'ftn_qb_epa': """Extract the table data from this screenshot. Return ONLY valid CSV with these exact headers, no explanation:
rank,player,team,games,dropbacks,epa,epa_per_dropback""",

    'pff_qb_grades': """Extract the table data from this screenshot. Return ONLY valid CSV with these exact headers, no explanation:
player,team,pass_grade,btt,btt_pct,twp,twp_pct""",

    'pff_oline': """Extract the table data from this screenshot. Return ONLY valid CSV with these exact headers, no explanation:
team,pass_block_wr,run_block_wr,overall_grade""",

    'pff_dline': """Extract the table data from this screenshot. Return ONLY valid CSV with these exact headers, no explanation:
team,pass_rush_wr,run_stop_wr,overall_grade""",

    'ngs_separation': """Extract the table data from this screenshot. Return ONLY valid CSV with these exact headers, no explanation:
player,team,separation,yac,yac_diff,catch_pct"""
}


def extract_table_with_claude(image_data: bytes, source_key: str) -> tuple:
    """
    Extract table data from image using Claude Vision API.
    Returns (DataFrame, error_message) tuple.
    """
    if not CLAUDE_VISION_AVAILABLE:
        return None, "ANTHROPIC_API_KEY not set. Add it to your .env file."

    # Get the prompt for this source
    prompt = EXTRACTION_PROMPTS.get(source_key,
        "Extract the table data from this screenshot. Return ONLY valid CSV format, no explanation.")

    # Convert image to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Determine media type (assume PNG, works for most screenshots)
    media_type = "image/png"

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        # Get the response text
        csv_text = message.content[0].text.strip()

        # Remove markdown code blocks if present
        if csv_text.startswith('```'):
            lines = csv_text.split('\n')
            csv_text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

        # Parse CSV into DataFrame
        try:
            df = pd.read_csv(StringIO(csv_text))
            return df, None
        except Exception as parse_error:
            return None, f"CSV parse error: {parse_error}\n\nRaw response:\n{csv_text[:500]}"

    except Exception as api_error:
        return None, f"Claude API error: {str(api_error)}"

# Page config
st.set_page_config(
    page_title="NFL Power Ratings Dashboard",
    page_icon="🏈",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/master_power_ratings.csv')
    # Override with fresh standings from nfl_data_py
    standings_path = Path('data/nfl_standings_2025.csv')
    if standings_path.exists():
        standings = pd.read_csv(standings_path)
        # Merge in updated wins/losses
        df = df.drop(columns=['wins', 'losses'], errors='ignore')
        df = df.merge(standings[['team', 'wins', 'losses']], on='team', how='left')
    return df

@st.cache_data
def load_edges():
    try:
        return pd.read_csv('outputs/edges.csv')
    except:
        return None

@st.cache_data
def load_allocations():
    try:
        return pd.read_csv('outputs/allocations.csv')
    except:
        return None

@st.cache_data
def load_factor_data():
    """Load all factor CSVs for detailed team profiles."""
    factor_data = {}
    try:
        factor_data['qb'] = pd.read_csv('data/processed/factor_qb_quality.csv')
    except:
        factor_data['qb'] = None
    try:
        factor_data['line'] = pd.read_csv('data/processed/factor_line_play.csv')
    except:
        factor_data['line'] = None
    try:
        factor_data['dline'] = pd.read_csv('data/processed/factor_dline.csv')
    except:
        factor_data['dline'] = None
    try:
        factor_data['receiving'] = pd.read_csv('data/processed/factor_ngs_receiving.csv')
    except:
        factor_data['receiving'] = None
    try:
        factor_data['situational'] = pd.read_csv('data/processed/factor_situational.csv')
    except:
        factor_data['situational'] = None
    try:
        factor_data['wins'] = pd.read_csv('data/processed/factor_wins.csv')
    except:
        factor_data['wins'] = None
    return factor_data

def get_rank(df, team, score_col, ascending=False):
    """Get rank of team in a dataframe by score column."""
    if df is None or team not in df['team'].values:
        return None
    sorted_df = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    try:
        return sorted_df[sorted_df['team'] == team].index[0] + 1
    except:
        return None

def generate_qb_assessment(epa, qb_score):
    """Generate one-sentence QB assessment."""
    if epa is None or pd.isna(epa):
        return "Limited data available for assessment."
    if epa >= 0.2:
        return "Elite efficiency - among the league's best at creating positive plays."
    elif epa >= 0.1:
        return "Above-average production with consistent positive contributions."
    elif epa >= 0.0:
        return "Steady performance that keeps the offense functional."
    elif epa >= -0.1:
        return "Below-average efficiency - room for improvement in execution."
    else:
        return "Struggling to generate positive plays - significant liability."

def generate_oline_assessment(pass_score, run_score):
    """Generate one-sentence O-Line assessment."""
    if pass_score is None or pd.isna(pass_score):
        return "Limited data available for assessment."
    avg_score = (pass_score + (run_score or 50)) / 2
    if avg_score >= 75:
        return "Dominant unit providing excellent protection and run lanes."
    elif avg_score >= 55:
        return "Solid group that handles most assignments effectively."
    elif avg_score >= 40:
        return "Average unit with occasional breakdowns in protection."
    else:
        return "Struggling to protect the QB and create running lanes."

def generate_pass_rush_assessment(pr_score):
    """Generate one-sentence pass rush assessment."""
    if pr_score is None or pd.isna(pr_score):
        return "Limited data available for assessment."
    if pr_score >= 75:
        return "Elite pass rush creating consistent pressure on opposing QBs."
    elif pr_score >= 55:
        return "Good pressure generation that disrupts opposing offenses."
    elif pr_score >= 40:
        return "Moderate pressure - can get home but not consistently."
    else:
        return "Lacking pass rush - gives QBs too much time in the pocket."

def generate_receiving_assessment(sep, yac_diff):
    """Generate one-sentence receiving corps assessment."""
    if sep is None or pd.isna(sep):
        return "Limited data available for assessment."
    if sep >= 3.2 and (yac_diff or 0) >= 0.5:
        return "Elite separation creators who excel after the catch."
    elif sep >= 3.0:
        return "Good route runners generating consistent separation."
    elif sep >= 2.7:
        return "Average separation - relies on contested catches."
    else:
        return "Struggling to get open - need improved route running."

def generate_situational_assessment(third_down, rz_rate):
    """Generate one-sentence situational assessment."""
    if third_down is None or pd.isna(third_down):
        return "Limited data available for assessment."
    if third_down >= 0.45 and (rz_rate or 0) >= 0.22:
        return "Excellent in clutch situations - converts when it matters."
    elif third_down >= 0.40:
        return "Solid situational execution with room for growth."
    elif third_down >= 0.35:
        return "Average in key situations - inconsistent results."
    else:
        return "Poor situational football - struggles in critical moments."

def generate_win_record_assessment(win_pct_score, wins, losses):
    """Generate one-sentence win record assessment."""
    if win_pct_score is None or pd.isna(win_pct_score):
        return "Limited data available for assessment."
    total_games = wins + losses if wins is not None and losses is not None else 0
    if win_pct_score >= 75:
        return f"Elite record ({wins}-{losses}) - proven winner this season."
    elif win_pct_score >= 60:
        return f"Strong record ({wins}-{losses}) - contender status confirmed."
    elif win_pct_score >= 45:
        return f"Average record ({wins}-{losses}) - could go either way."
    elif win_pct_score >= 30:
        return f"Below average ({wins}-{losses}) - struggling to find wins."
    else:
        return f"Poor record ({wins}-{losses}) - in rebuilding mode."

def generate_detailed_profile(team, team_data, factor_data, edges_df, allocs_df, overall_rank):
    """Generate detailed factor-by-factor team profile."""
    profile_parts = []

    # Team header with overall rank
    team_name = team_data.get('name', team)
    profile_parts.append(f"**{team_name}** - Overall Rank: **#{overall_rank}**")
    profile_parts.append("")

    # QB Quality Section
    qb_df = factor_data.get('qb')
    if qb_df is not None and team in qb_df['team'].values:
        qb_row = qb_df[qb_df['team'] == team].iloc[0]
        qb_name = qb_row.get('primary_qb', 'Unknown')
        epa = qb_row.get('epa_dropback', None)
        qb_score = qb_row.get('qb_quality_score', None)
        qb_rank = get_rank(qb_df, team, 'qb_quality_score', ascending=False)

        profile_parts.append(f"**QB Quality** (Rank #{qb_rank}, Score: {qb_score:.1f}):")
        if epa is not None and not pd.isna(epa):
            profile_parts.append(f"- {qb_name}: {epa:.2f} EPA/dropback (Rank #{qb_rank} among QBs)")
        else:
            profile_parts.append(f"- {qb_name}: Data limited")
        profile_parts.append(f"- {generate_qb_assessment(epa, qb_score)}")
        profile_parts.append("")

    # Offensive Line Section
    line_df = factor_data.get('line')
    if line_df is not None and team in line_df['team'].values:
        line_row = line_df[line_df['team'] == team].iloc[0]
        pass_score = line_row.get('ol_pass_score', None)
        run_score = line_row.get('run_block_score', None)
        line_score = line_row.get('line_play_score', None)
        line_rank = get_rank(line_df, team, 'line_play_score', ascending=False)

        profile_parts.append(f"**Offensive Line** (Rank #{line_rank}, Score: {line_score:.1f}):")
        if pass_score is not None:
            profile_parts.append(f"- Pass Block Score: {pass_score:.1f}")
        if run_score is not None:
            profile_parts.append(f"- Run Block Score: {run_score:.1f}")
        profile_parts.append(f"- {generate_oline_assessment(pass_score, run_score)}")
        profile_parts.append("")

    # Pass Rush Section
    dline_df = factor_data.get('dline')
    if dline_df is not None and team in dline_df['team'].values:
        dline_row = dline_df[dline_df['team'] == team].iloc[0]
        pr_score = dline_row.get('pass_rush_score', None)
        dline_score = dline_row.get('dline_composite_score', None)
        sack_rate = dline_row.get('sack_rate', None)
        dline_rank = get_rank(dline_df, team, 'dline_composite_score', ascending=False)

        profile_parts.append(f"**Pass Rush** (Rank #{dline_rank}, Score: {dline_score:.1f}):")
        if sack_rate is not None:
            profile_parts.append(f"- Sack Rate: {sack_rate*100:.1f}%")
        profile_parts.append(f"- {generate_pass_rush_assessment(pr_score)}")
        profile_parts.append("")

    # Receiving Corps Section
    rec_df = factor_data.get('receiving')
    if rec_df is not None and team in rec_df['team'].values:
        rec_row = rec_df[rec_df['team'] == team].iloc[0]
        sep = rec_row.get('separation', None)
        yac_diff = rec_row.get('yac_diff', None)
        rec_score = rec_row.get('receiving_score', None)
        rec_rank = get_rank(rec_df, team, 'receiving_score', ascending=False)

        profile_parts.append(f"**Receiving Corps** (Rank #{rec_rank}, Score: {rec_score:.1f}):")
        if sep is not None:
            profile_parts.append(f"- Avg Separation: {sep:.1f} yards")
        if yac_diff is not None:
            sign = "+" if yac_diff >= 0 else ""
            profile_parts.append(f"- YAC vs Expected: {sign}{yac_diff:.1f}")
        profile_parts.append(f"- {generate_receiving_assessment(sep, yac_diff)}")
        profile_parts.append("")

    # Situational Section
    sit_df = factor_data.get('situational')
    if sit_df is not None and team in sit_df['team'].values:
        sit_row = sit_df[sit_df['team'] == team].iloc[0]
        third = sit_row.get('third_down_success', None)
        rz = sit_row.get('red_zone_td_rate', None)
        sit_score = sit_row.get('situational_score', None)
        sit_rank = get_rank(sit_df, team, 'situational_score', ascending=False)

        profile_parts.append(f"**Situational** (Rank #{sit_rank}, Score: {sit_score:.1f}):")
        if third is not None:
            profile_parts.append(f"- 3rd Down Success: {third*100:.1f}%")
        if rz is not None:
            profile_parts.append(f"- Red Zone TD Rate: {rz*100:.1f}%")
        profile_parts.append(f"- {generate_situational_assessment(third, rz)}")
        profile_parts.append("")

    # Win Record Section
    wins_df = factor_data.get('wins')
    if wins_df is not None and team in wins_df['team'].values:
        wins_row = wins_df[wins_df['team'] == team].iloc[0]
        win_pct_score = wins_row.get('win_pct_score', None)
        wins = wins_row.get('wins', None)
        losses = wins_row.get('losses', None)
        win_pct = wins_row.get('win_pct', None)
        wins_rank = get_rank(wins_df, team, 'win_pct_score', ascending=False)

        profile_parts.append(f"**Win Record** (Rank #{wins_rank}, Score: {win_pct_score:.1f}):")
        if wins is not None and losses is not None:
            profile_parts.append(f"- Record: {int(wins)}-{int(losses)} ({win_pct*100:.1f}%)")
        profile_parts.append(f"- {generate_win_record_assessment(win_pct_score, wins, losses)}")
        profile_parts.append("")

    # Edge vs Market
    if edges_df is not None and team in edges_df['team'].values:
        edge_row = edges_df[edges_df['team'] == team].iloc[0]
        edge_pct = edge_row.get('edge_pct', 0)
        if edge_pct > 0:
            profile_parts.append(f"**EDGE VS MARKET:** +{edge_pct:.1f}% undervalued")
        else:
            profile_parts.append(f"**EDGE VS MARKET:** {edge_pct:.1f}% overvalued")
    else:
        profile_parts.append("**EDGE VS MARKET:** N/A (no market data)")

    # Kelly Allocation
    if allocs_df is not None and team in allocs_df['team'].values:
        alloc_row = allocs_df[allocs_df['team'] == team].iloc[0]
        kelly = alloc_row.get('kelly_fraction', 0) * 100
        profile_parts.append(f"**KELLY ALLOCATION:** {kelly:.1f}% of bankroll")
    else:
        profile_parts.append("**KELLY ALLOCATION:** 0% (no edge)")

    return "\n".join(profile_parts)

# Ensure upload directory exists
UPLOAD_DIR = Path('data/uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Data source configuration
DATA_SOURCES = {
    'ftn_qb_epa': {
        'name': 'FTN QB EPA',
        'description': 'QB EPA per dropback rankings',
        'target_csv': 'data/ftn_epa_2025.csv',
        'icon': '🎯'
    },
    'pff_qb_grades': {
        'name': 'PFF QB Grades',
        'description': 'BTT, TWP, TWP%, Pass Grade',
        'target_csv': 'data/stats/pff_qb_grades.csv',
        'icon': '🏈'
    },
    'pff_oline': {
        'name': 'PFF O-Line Win Rate',
        'description': 'Pass Block WR, Run Block WR',
        'target_csv': 'data/stats/pff_oline_grades.csv',
        'icon': '🛡️'
    },
    'pff_dline': {
        'name': 'PFF Pass Rush Win Rate',
        'description': 'Pass Rush WR, Run Stop WR',
        'target_csv': 'data/stats/pff_dline_grades.csv',
        'icon': '💪'
    },
    'ngs_separation': {
        'name': 'NGS Receiver Separation',
        'description': 'Separation, YAC, Catch %',
        'target_csv': 'data/stats/ngs_separation.csv',
        'icon': '📡'
    }
}

def save_uploaded_image(uploaded_file, source_key: str, index: int = 0) -> str:
    """Save uploaded image to data/uploads/ with timestamp."""
    import time
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # Add milliseconds and index to ensure unique filenames for batch uploads
    ms_suffix = f"_{int(time.time() * 1000) % 1000:03d}_{index:02d}"
    file_ext = Path(uploaded_file.name).suffix.lower()
    filename = f"{source_key}_{timestamp}{ms_suffix}{file_ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    return str(filepath)

def get_recent_uploads(source_key: str, limit: int = 10) -> list:
    """Get most recent uploads for a source."""
    pattern = f"{source_key}_*"
    files = sorted(UPLOAD_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:limit]

def delete_uploaded_file(filepath: Path) -> bool:
    """Delete an uploaded file."""
    try:
        if filepath.exists():
            filepath.unlink()
            return True
    except Exception:
        pass
    return False

# Load the data
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:

    # Load edge and allocation data
    edges_df = load_edges()
    allocs_df = load_allocations()
    factor_data = load_factor_data()

    # Title
    st.title("🏈 NFL Power Ratings Dashboard")
    st.markdown("**Compare 6 different weight distributions across all 32 teams**")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_options = {
        'main_power': 'MAIN MODEL',
        'qb_heavy_power': 'QB HEAVY',
        'efficiency_heavy_power': 'EFFICIENCY HEAVY',
        'trenches_power': 'TRENCHES',
        'process_power': 'PROCESS',
        'skill_only_power': 'SKILL ONLY'
    }
    
    selected_model = st.sidebar.selectbox(
        "Primary Model for Rankings",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Sort dataframe by selected model
    df_sorted = df.sort_values(selected_model, ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Rankings", "🔍 Team Profiles", "📈 Model Comparison", "📋 Factor Breakdown", "💰 Edge Detection", "📊 Kelly Allocation", "📤 Data Updates"])
    
    # TAB 1: Rankings
    with tab1:
        st.header(f"Rankings by {model_options[selected_model]}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔝 Top 10 Teams")
            top10 = df_sorted.head(10)[['Rank', 'team', 'name', 'wins', 'losses', selected_model]].copy()
            top10.columns = ['Rank', 'Team', 'Name', 'W', 'L', 'Power Rating']
            top10['Power Rating'] = top10['Power Rating'].round(1)
            st.dataframe(top10, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("⬇️ Bottom 10 Teams")
            bottom10 = df_sorted.tail(10)[['Rank', 'team', 'name', 'wins', 'losses', selected_model]].copy()
            bottom10.columns = ['Rank', 'Team', 'Name', 'W', 'L', 'Power Rating']
            bottom10['Power Rating'] = bottom10['Power Rating'].round(1)
            st.dataframe(bottom10, hide_index=True, use_container_width=True)
        
        # Full rankings
        st.subheader("📋 Full Rankings - All 32 Teams")
        full_rankings = df_sorted[['Rank', 'team', 'name', 'wins', 'losses', selected_model]].copy()
        full_rankings.columns = ['Rank', 'Team', 'Name', 'W', 'L', 'Power Rating']
        full_rankings['Power Rating'] = full_rankings['Power Rating'].round(1)
        st.dataframe(full_rankings, hide_index=True, use_container_width=True, height=600)
    
    # TAB 2: Team Profiles
    with tab2:
        st.header("🔍 Team Profile Explorer")
        
        # Team selector
        team_list = df_sorted['team'].tolist()
        selected_team = st.selectbox("Select a Team", team_list, format_func=lambda x: f"{x} - {df[df['team']==x]['name'].values[0]}")
        
        team_data = df[df['team'] == selected_team].iloc[0]
        
        # Team header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"## {team_data['name']}")
        with col2:
            st.metric("Record", f"{int(team_data['wins'])}-{int(team_data['losses'])}")
        with col3:
            main_rank = df_sorted[df_sorted['team'] == selected_team]['Rank'].values[0]
            st.metric(f"{model_options[selected_model]} Rank", f"#{main_rank}")
        
        st.divider()

        # Team Analysis Summary - Detailed Factor Breakdown
        st.subheader("📝 Detailed Team Analysis")

        # Generate and display detailed profile
        overall_rank = df_sorted[df_sorted['team'] == selected_team]['Rank'].values[0]
        detailed_profile = generate_detailed_profile(
            selected_team,
            team_data.to_dict(),
            factor_data,
            edges_df,
            allocs_df,
            overall_rank
        )
        st.markdown(detailed_profile)

        st.divider()

        # Factor scores
        st.subheader("📊 Factor Scores (0-100 scale)")
        
        factor_cols = ['win_pct_score', 'efficiency_score', 'oline_score', 'situational_score',
                       'dline_score', 'ngs_receiving_score', 'qb_quality_score']
        factor_names = ['Win Record', 'Efficiency', 'O-Line', 'Situational', 'D-Line', 'NGS Receiving', 'QB Quality']

        cols = st.columns(4)
        for i, (col_name, factor_name) in enumerate(zip(factor_cols, factor_names)):
            with cols[i % 4]:
                value = team_data.get(col_name, None)
                if value is not None and pd.notna(value):
                    # Color code based on score
                    if value >= 70:
                        delta = "Elite"
                    elif value >= 55:
                        delta = "Above Avg"
                    elif value >= 45:
                        delta = "Average"
                    elif value >= 30:
                        delta = "Below Avg"
                    else:
                        delta = "Poor"
                    st.metric(factor_name, f"{value:.1f}", delta)
                else:
                    st.metric(factor_name, "N/A")
        
        st.divider()
        
        # Power ratings across all models
        st.subheader("⚖️ Power Ratings Across All Models")
        
        power_cols = ['main_power', 'qb_heavy_power', 'efficiency_heavy_power',
                      'trenches_power', 'process_power', 'skill_only_power']

        cols = st.columns(6)
        for i, (col_name, model_name) in enumerate(zip(power_cols, model_options.values())):
            with cols[i]:
                value = team_data[col_name]
                # Get rank for this model
                model_sorted = df.sort_values(col_name, ascending=False).reset_index(drop=True)
                model_sorted['rank'] = range(1, len(model_sorted) + 1)
                rank = model_sorted[model_sorted['team'] == selected_team]['rank'].values[0]
                st.metric(model_name, f"{value:.1f}", f"#{rank}")
        
        # Strengths and weaknesses
        st.divider()
        st.subheader("💪 Strengths & Weaknesses")
        
        factor_scores = {name: team_data[col] for col, name in zip(factor_cols, factor_names) if pd.notna(team_data[col])}
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 3 Strengths:**")
            for name, score in sorted_factors[:3]:
                st.markdown(f"- {name}: **{score:.1f}**")
        
        with col2:
            st.markdown("**Bottom 3 Weaknesses:**")
            for name, score in sorted_factors[-3:]:
                st.markdown(f"- {name}: **{score:.1f}**")
    
    # TAB 3: Model Comparison
    with tab3:
        st.header("📈 Model Comparison")
        
        # Cross-model correlation
        st.subheader("🔗 Model Correlations with Wins")
        
        power_cols = ['main_power', 'qb_heavy_power', 'efficiency_heavy_power',
                      'trenches_power', 'process_power', 'skill_only_power']

        correlations = {}
        for col in power_cols:
            corr = df['wins'].corr(df[col])
            correlations[model_options[col]] = corr
        
        corr_df = pd.DataFrame({
            'Model': correlations.keys(),
            'Correlation with Wins': [f"{v:.3f}" for v in correlations.values()]
        })
        st.dataframe(corr_df, hide_index=True)
        
        st.divider()
        
        # Biggest movers between models
        st.subheader("🔄 Biggest Rank Changes Between Models")
        
        model1 = st.selectbox("Model 1", list(model_options.keys()), index=0, format_func=lambda x: model_options[x], key='m1')
        model2 = st.selectbox("Model 2", list(model_options.keys()), index=4, format_func=lambda x: model_options[x], key='m2')
        
        if model1 != model2:
            # Get ranks for both models
            df_m1 = df.sort_values(model1, ascending=False).reset_index(drop=True)
            df_m1['rank_m1'] = range(1, len(df_m1) + 1)
            
            df_m2 = df.sort_values(model2, ascending=False).reset_index(drop=True)
            df_m2['rank_m2'] = range(1, len(df_m2) + 1)
            
            # Merge
            rank_compare = df_m1[['team', 'name', 'rank_m1']].merge(
                df_m2[['team', 'rank_m2']], on='team'
            )
            rank_compare['rank_change'] = rank_compare['rank_m1'] - rank_compare['rank_m2']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Biggest Risers ({model_options[model1]} → {model_options[model2]}):**")
                risers = rank_compare.nlargest(5, 'rank_change')[['team', 'name', 'rank_m1', 'rank_m2', 'rank_change']]
                risers.columns = ['Team', 'Name', f'{model_options[model1]} Rank', f'{model_options[model2]} Rank', 'Change']
                st.dataframe(risers, hide_index=True)
            
            with col2:
                st.markdown(f"**Biggest Fallers ({model_options[model1]} → {model_options[model2]}):**")
                fallers = rank_compare.nsmallest(5, 'rank_change')[['team', 'name', 'rank_m1', 'rank_m2', 'rank_change']]
                fallers.columns = ['Team', 'Name', f'{model_options[model1]} Rank', f'{model_options[model2]} Rank', 'Change']
                st.dataframe(fallers, hide_index=True)
        
        st.divider()
        
        # Model agreement/disagreement
        st.subheader("🎯 Model Agreement Analysis")
        
        # Calculate standard deviation of ranks across models for each team
        rank_columns = []
        for col in power_cols:
            df_temp = df.sort_values(col, ascending=False).reset_index(drop=True)
            df_temp[f'{col}_rank'] = range(1, len(df_temp) + 1)
            rank_columns.append(df_temp[['team', f'{col}_rank']])
        
        rank_df = rank_columns[0]
        for rc in rank_columns[1:]:
            rank_df = rank_df.merge(rc, on='team')
        
        rank_cols = [c for c in rank_df.columns if '_rank' in c]
        rank_df['rank_std'] = rank_df[rank_cols].std(axis=1)
        rank_df['rank_mean'] = rank_df[rank_cols].mean(axis=1)
        rank_df = rank_df.merge(df[['team', 'name']], on='team')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Consistent Rankings (All Models Agree):**")
            consistent = rank_df.nsmallest(5, 'rank_std')[['team', 'name', 'rank_mean', 'rank_std']]
            consistent.columns = ['Team', 'Name', 'Avg Rank', 'Rank Std Dev']
            consistent['Avg Rank'] = consistent['Avg Rank'].round(1)
            consistent['Rank Std Dev'] = consistent['Rank Std Dev'].round(2)
            st.dataframe(consistent, hide_index=True)
        
        with col2:
            st.markdown("**Most Volatile Rankings (Models Disagree):**")
            volatile = rank_df.nlargest(5, 'rank_std')[['team', 'name', 'rank_mean', 'rank_std']]
            volatile.columns = ['Team', 'Name', 'Avg Rank', 'Rank Std Dev']
            volatile['Avg Rank'] = volatile['Avg Rank'].round(1)
            volatile['Rank Std Dev'] = volatile['Rank Std Dev'].round(2)
            st.dataframe(volatile, hide_index=True)
    
    # TAB 4: Factor Breakdown
    with tab4:
        st.header("📋 Factor Breakdown")
        
        st.subheader("🏆 Factor Leaders")
        
        factor_cols = ['win_pct_score', 'efficiency_score', 'oline_score', 'situational_score',
                       'dline_score', 'ngs_receiving_score', 'qb_quality_score']
        factor_names = ['Win Record', 'Efficiency', 'O-Line', 'Situational', 'D-Line', 'NGS Receiving', 'QB Quality']

        for col_name, factor_name in zip(factor_cols, factor_names):
            with st.expander(f"**{factor_name}** - Top 5"):
                top5 = df.nlargest(5, col_name)[['team', 'name', 'wins', 'losses', col_name]].copy()
                top5.columns = ['Team', 'Name', 'W', 'L', 'Score']
                top5['Score'] = top5['Score'].round(1)
                st.dataframe(top5, hide_index=True, use_container_width=True)
        
        st.divider()
        
        st.subheader("📊 Factor Correlations with Wins")
        
        correlations = {}
        for col, name in zip(factor_cols, factor_names):
            valid_df = df.dropna(subset=[col])
            corr = valid_df['wins'].corr(valid_df[col])
            correlations[name] = corr
        
        corr_df = pd.DataFrame({
            'Factor': correlations.keys(),
            'Correlation': [f"{v:.3f}" for v in correlations.values()]
        }).sort_values('Correlation', ascending=False)
        
        st.dataframe(corr_df, hide_index=True)
        
        st.markdown("""
        **Interpretation:**
        - **Positive correlation**: Higher factor score → More wins
        - Correlations > 0.3 are considered moderately strong for sports data
        - Win Record factor has perfect correlation by design (derived from wins)
        """)

    # TAB 5: Edge Detection
    with tab5:
        st.header("💰 Edge Detection vs Polymarket")
        st.markdown("**Model probability vs market implied probability - positive edge = model sees more value**")

        if edges_df is not None and len(edges_df) > 0:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                positive_edges = len(edges_df[edges_df['edge'] > 0])
                st.metric("Positive Edges", positive_edges)
            with col2:
                max_edge = edges_df['edge_pct'].max()
                st.metric("Max Edge", f"{max_edge:.1f}%")
            with col3:
                actionable = len(edges_df[edges_df['edge'] >= 0.02])
                st.metric("Actionable (>2%)", actionable)
            with col4:
                avg_edge = edges_df[edges_df['edge'] > 0]['edge_pct'].mean()
                st.metric("Avg Positive Edge", f"{avg_edge:.1f}%")

            st.divider()

            # Betting opportunities (positive edge)
            st.subheader("🎯 Betting Opportunities (Positive Edge)")
            positive_df = edges_df[edges_df['edge'] > 0].copy()
            if len(positive_df) > 0:
                display_cols = ['team', 'model_prob', 'implied_prob', 'edge_pct', 'ev_pct']
                display_df = positive_df[display_cols].copy()
                display_df.columns = ['Team', 'Model Prob', 'Market Prob', 'Edge %', 'EV %']
                display_df['Model Prob'] = (display_df['Model Prob'] * 100).round(1).astype(str) + '%'
                display_df['Market Prob'] = (display_df['Market Prob'] * 100).round(1).astype(str) + '%'
                display_df['Edge %'] = display_df['Edge %'].round(1).astype(str) + '%'
                display_df['EV %'] = display_df['EV %'].round(1).astype(str) + '%'
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("No positive edges found in current data")

            st.divider()

            # Full edge table
            st.subheader("📊 All Teams - Edge Analysis")
            full_df = edges_df[['team', 'model_prob', 'implied_prob', 'edge_pct', 'ev_pct', 'has_edge']].copy()
            full_df.columns = ['Team', 'Model Prob', 'Market Prob', 'Edge %', 'EV %', 'Has Edge']
            full_df['Model Prob'] = (full_df['Model Prob'] * 100).round(2).astype(str) + '%'
            full_df['Market Prob'] = (full_df['Market Prob'] * 100).round(2).astype(str) + '%'
            full_df['Edge %'] = full_df['Edge %'].round(2)
            full_df['EV %'] = full_df['EV %'].round(1)
            st.dataframe(full_df, hide_index=True, use_container_width=True, height=400)
        else:
            st.warning("Edge data not available. Run edge_detector.py first.")

    # TAB 6: Kelly Allocation
    with tab6:
        st.header("📊 Kelly Criterion Allocation")
        st.markdown("**Optimal position sizing using fractional Kelly criterion (25% Kelly)**")

        if allocs_df is not None and len(allocs_df) > 0:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_positions = len(allocs_df)
                st.metric("Total Positions", total_positions)
            with col2:
                total_allocated = allocs_df['final_amount'].sum()
                st.metric("Total Allocated", f"${total_allocated:.2f}")
            with col3:
                total_expected = allocs_df['expected_profit'].sum()
                st.metric("Expected Profit", f"${total_expected:.2f}")
            with col4:
                expected_roi = (total_expected / total_allocated * 100) if total_allocated > 0 else 0
                st.metric("Expected ROI", f"{expected_roi:.1f}%")

            st.divider()

            # Allocation table
            st.subheader("💵 Recommended Allocations")
            alloc_display = allocs_df[['team', 'model_prob', 'market_price', 'edge_pct', 'final_amount', 'expected_profit']].copy()
            alloc_display.columns = ['Team', 'Model Prob', 'Market Price', 'Edge %', 'Allocation ($)', 'Expected Profit ($)']
            alloc_display['Model Prob'] = (alloc_display['Model Prob'] * 100).round(1).astype(str) + '%'
            alloc_display['Market Price'] = (alloc_display['Market Price'] * 100).round(1).astype(str) + '%'
            alloc_display['Edge %'] = alloc_display['Edge %'].round(1).astype(str) + '%'
            alloc_display['Allocation ($)'] = alloc_display['Allocation ($)'].round(2)
            alloc_display['Expected Profit ($)'] = alloc_display['Expected Profit ($)'].round(2)
            st.dataframe(alloc_display, hide_index=True, use_container_width=True)

            st.divider()

            # Kelly explanation
            st.subheader("📖 Kelly Criterion Explained")
            st.markdown("""
            **Kelly Formula:** f* = (bp - q) / b

            Where:
            - **f*** = fraction of bankroll to bet
            - **b** = odds (payout ratio - 1)
            - **p** = probability of winning (model probability)
            - **q** = probability of losing (1 - p)

            **Fractional Kelly (25%):** We use quarter Kelly to reduce volatility while maintaining positive expected value.
            This means betting 25% of what full Kelly recommends.
            """)
        else:
            st.warning("Allocation data not available. Run position_sizer.py first.")

    # TAB 7: Data Updates
    with tab7:
        st.header("📤 Weekly Data Updates")
        st.markdown("**Upload screenshots from data sources to update model inputs**")

        # Claude Vision status indicator
        if CLAUDE_VISION_AVAILABLE:
            st.success("✅ Claude Vision Available (Anthropic API)")
        else:
            st.error("❌ Claude Vision Not Available. Set ANTHROPIC_API_KEY in your .env file.")

        st.markdown("---")

        # Create upload sections for each data source
        for source_key, source_info in DATA_SOURCES.items():
            with st.expander(f"{source_info['icon']} {source_info['name']}", expanded=True):
                st.markdown(f"*{source_info['description']}*")
                st.markdown(f"Target CSV: `{source_info['target_csv']}`")

                # Initialize session state for this source
                if f'ocr_result_{source_key}' not in st.session_state:
                    st.session_state[f'ocr_result_{source_key}'] = None
                if f'ocr_file_{source_key}' not in st.session_state:
                    st.session_state[f'ocr_file_{source_key}'] = None
                if f'ocr_saved_path_{source_key}' not in st.session_state:
                    st.session_state[f'ocr_saved_path_{source_key}'] = None

                # Multi-file uploader
                uploaded_files = st.file_uploader(
                    f"Upload {source_info['name']} screenshots (multiple allowed)",
                    type=['png', 'jpg', 'jpeg', 'webp'],
                    key=f"uploader_{source_key}",
                    accept_multiple_files=True
                )

                # Handle uploaded files - always save, optionally run Claude Vision
                if uploaded_files and st.session_state[f'ocr_result_{source_key}'] is None:
                    # Save all uploaded files
                    for idx, uploaded_file in enumerate(uploaded_files):
                        saved_path = save_uploaded_image(uploaded_file, source_key, idx)
                        if idx == 0:
                            st.session_state[f'ocr_saved_path_{source_key}'] = saved_path
                            st.session_state[f'ocr_file_{source_key}'] = uploaded_file.name

                    st.success(f"Saved {len(uploaded_files)} file(s) to data/uploads/")

                    # Auto-trigger Claude Vision extraction if available
                    if CLAUDE_VISION_AVAILABLE:
                        current_file = uploaded_files[0]
                        with st.spinner(f"🔍 Extracting data with Claude Vision from {current_file.name}..."):
                            current_file.seek(0)
                            image_bytes = current_file.read()
                            df, error = extract_table_with_claude(image_bytes, source_key)
                            if df is not None:
                                st.session_state[f'ocr_result_{source_key}'] = df
                                st.session_state[f'ocr_error_{source_key}'] = None
                            else:
                                st.session_state[f'ocr_result_{source_key}'] = 'error'
                                st.session_state[f'ocr_error_{source_key}'] = error
                    else:
                        st.info("Claude Vision not available. Files saved - you can ask Claude to extract them manually.")

                # Show OCR result preview and confirm/cancel buttons
                if st.session_state[f'ocr_result_{source_key}'] is not None:
                    result = st.session_state[f'ocr_result_{source_key}']
                    current_file = st.session_state[f'ocr_file_{source_key}']

                    st.markdown(f"**Extracted from:** `{current_file}`")

                    if result != 'error' and len(result) > 0:
                        st.markdown(f"**Preview (first 5 rows of {len(result)} total):**")
                        st.dataframe(result.head(5), use_container_width=True)
                        st.markdown(f"**Columns found:** {list(result.columns)}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Confirm Save", key=f"confirm_{source_key}", type="primary"):
                                # Save to target CSV
                                target_path = Path(source_info['target_csv'])
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                result.to_csv(target_path, index=False)
                                st.success(f"Saved {len(result)} rows to `{source_info['target_csv']}`")

                                # Clear current result
                                st.session_state[f'ocr_result_{source_key}'] = None
                                st.session_state[f'ocr_file_{source_key}'] = None
                                st.session_state[f'ocr_saved_path_{source_key}'] = None
                                st.session_state[f'ocr_error_{source_key}'] = None
                                st.rerun()

                        with col2:
                            if st.button("❌ Cancel", key=f"cancel_{source_key}"):
                                st.session_state[f'ocr_result_{source_key}'] = None
                                st.session_state[f'ocr_file_{source_key}'] = None
                                st.session_state[f'ocr_saved_path_{source_key}'] = None
                                st.session_state[f'ocr_error_{source_key}'] = None
                                st.warning("Extraction cancelled. File saved in uploads but not converted.")
                                st.rerun()
                    else:
                        error_msg = st.session_state.get(f'ocr_error_{source_key}', 'Unknown error')
                        st.error(f"❌ Could not extract table from image: {error_msg}")
                        if st.button("Dismiss", key=f"dismiss_{source_key}"):
                            st.session_state[f'ocr_result_{source_key}'] = None
                            st.session_state[f'ocr_file_{source_key}'] = None
                            st.session_state[f'ocr_saved_path_{source_key}'] = None
                            st.session_state[f'ocr_error_{source_key}'] = None
                            st.rerun()

                st.markdown("---")

                # Recent uploads for re-processing
                st.markdown("**Recent Uploads (click to re-process):**")
                recent = get_recent_uploads(source_key, limit=10)

                # Handle pending delete action
                if 'delete_file_path' in st.session_state and st.session_state['delete_file_path']:
                    file_to_delete = Path(st.session_state['delete_file_path'])
                    if delete_uploaded_file(file_to_delete):
                        st.toast(f"Deleted {file_to_delete.name}")
                    st.session_state['delete_file_path'] = None
                    st.rerun()

                if recent:
                    for idx, f in enumerate(recent[:5]):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.caption(f.name)
                        with col2:
                            if CLAUDE_VISION_AVAILABLE:
                                if st.button("🔄 Re-OCR", key=f"reocr_{source_key}_{idx}"):
                                    with st.spinner(f"🔍 Extracting data from {f.name}..."):
                                        df, error = extract_table_with_claude(str(f), source_key)
                                        st.session_state[f'ocr_result_{source_key}'] = df if df is not None else 'error'
                                        st.session_state[f'ocr_file_{source_key}'] = f.name
                                        st.session_state[f'ocr_saved_path_{source_key}'] = str(f)
                                        st.rerun()
                        with col3:
                            st.button("🗑️", key=f"del_{source_key}_{idx}",
                                     on_click=lambda fp=str(f): st.session_state.update({'delete_file_path': fp}))
                else:
                    st.info("No files uploaded yet for this source.")

        st.markdown("---")
        st.subheader("📁 All Uploaded Files")

        # List all files in upload directory
        all_uploads = [f for f in UPLOAD_DIR.glob("*") if f.is_file()]
        all_uploads = sorted(all_uploads, key=lambda x: x.stat().st_mtime, reverse=True)

        if all_uploads:
            upload_data = []
            for f in all_uploads:
                source = f.name.split('_')[0] if '_' in f.name else 'unknown'
                upload_data.append({
                    'Source': source,
                    'Filename': f.name,
                    'Size (KB)': round(f.stat().st_size / 1024, 1),
                    'Uploaded': datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
            st.dataframe(pd.DataFrame(upload_data), hide_index=True, use_container_width=True)

            if st.button("Clear All Uploads", type="secondary"):
                for f in all_uploads:
                    delete_uploaded_file(f)
                st.success("All files deleted!")
                st.rerun()
        else:
            st.info("No files uploaded yet. Use the upload sections above to add data screenshots.")

    # Footer
    st.divider()
    st.markdown("*NFL Power Ratings Dashboard | Data from 2025 Season | Generated by NFL Polymarket Model*")


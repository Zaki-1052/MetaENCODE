# scripts/analyze_commit_time.py
"""Analyze git commit timestamps to estimate active coding time.

Approach:
- Parse commits keeping original local time (don't convert to UTC)
- Group by author separately (human vs AI-assisted work)
- Session = commits within 90 min gap
- Time estimate = span from first to last commit + small end buffer
- No inflated "start buffer" - first commit time is the start
"""

from datetime import datetime, timedelta
from pathlib import Path
import re


def parse_timestamp_local(ts_str):
    """Parse git timestamp, return local datetime and timezone offset.

    Returns tuple of (local_datetime, tz_offset_hours) to preserve original time.
    """
    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([+-]\d{4})', ts_str)
    if not match:
        return None, None

    dt_str, tz_str = match.groups()
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    tz_hours = int(tz_str[:3])

    return dt, tz_hours


def parse_timestamp_utc(ts_str):
    """Parse git timestamp to UTC for sorting/comparison."""
    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([+-]\d{4})', ts_str)
    if not match:
        return None

    dt_str, tz_str = match.groups()
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    tz_hours = int(tz_str[:3])
    tz_mins = int(tz_str[0] + tz_str[3:]) if len(tz_str) > 3 else 0

    # Convert to UTC
    dt_utc = dt - timedelta(hours=tz_hours, minutes=tz_mins)
    return dt_utc


def load_commits(filepath):
    """Load and parse commits from file."""
    parsed = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue

            ts_utc = parse_timestamp_utc(parts[0])
            ts_local, tz_offset = parse_timestamp_local(parts[0])

            if ts_utc and ts_local:
                parsed.append({
                    'ts_utc': ts_utc,
                    'ts_local': ts_local,
                    'tz_offset': tz_offset,
                    'msg': parts[1],
                    'author': parts[2],
                    'raw': parts[0]
                })

    # Sort by UTC time for correct chronological order
    parsed.sort(key=lambda x: x['ts_utc'])
    return parsed


def identify_sessions_by_author(commits, gap_minutes=90):
    """Group commits into sessions, separated by author.

    Human commits and Claude commits are analyzed separately since
    Claude commits represent AI-generated code that required human
    prompting but not continuous human coding.
    """
    human_commits = [c for c in commits if c['author'] != 'Claude']
    claude_commits = [c for c in commits if c['author'] == 'Claude']

    def group_into_sessions(commit_list, gap_min):
        if not commit_list:
            return []

        session_gap = timedelta(minutes=gap_min)
        sessions = []
        current_session = [commit_list[0]]

        for i in range(1, len(commit_list)):
            gap = commit_list[i]['ts_utc'] - commit_list[i-1]['ts_utc']
            if gap <= session_gap:
                current_session.append(commit_list[i])
            else:
                sessions.append(current_session)
                current_session = [commit_list[i]]

        sessions.append(current_session)
        return sessions

    return {
        'human': group_into_sessions(human_commits, gap_minutes),
        'claude': group_into_sessions(claude_commits, gap_minutes)
    }


def estimate_session_time(session, end_buffer_min=10, single_commit_min=20):
    """Estimate active coding time for a session.

    Conservative approach:
    - Single commit: 20 min estimate
    - Multiple commits: span + small end buffer only
    - No inflated start buffer (first commit IS the start)
    """
    if len(session) == 1:
        return timedelta(minutes=single_commit_min)

    span = session[-1]['ts_utc'] - session[0]['ts_utc']
    return span + timedelta(minutes=end_buffer_min)


def format_local_time(commit):
    """Format commit time in its original local timezone."""
    tz_str = f"{commit['tz_offset']:+03d}00"
    return f"{commit['ts_local'].strftime('%H:%M')} ({tz_str})"


def analyze_commits(commits_file):
    """Main analysis function."""
    commits = load_commits(commits_file)
    sessions_by_author = identify_sessions_by_author(commits)

    human_sessions = sessions_by_author['human']
    claude_sessions = sessions_by_author['claude']

    print("=" * 80)
    print("COMMIT TIME ANALYSIS")
    print("=" * 80)
    print(f"\nTotal commits: {len(commits)}")
    print(f"Human commits: {sum(len(s) for s in human_sessions)}")
    print(f"Claude (AI-assisted) commits: {sum(len(s) for s in claude_sessions)}")
    print(f"Session gap threshold: 90 minutes")

    # Analyze human sessions
    print("\n" + "=" * 80)
    print("HUMAN WORK SESSIONS (Your actual coding time)")
    print("=" * 80)

    total_human_time = timedelta()

    for i, session in enumerate(human_sessions, 1):
        first = session[0]
        last = session[-1]
        span = last['ts_utc'] - first['ts_utc']
        active = estimate_session_time(session)
        total_human_time += active

        print(f"\n--- Session {i} ---")
        print(f"Date: {first['ts_local'].strftime('%Y-%m-%d')}")
        print(f"Start: {format_local_time(first)} - {first['msg'][:45]}")
        print(f"End:   {format_local_time(last)} - {last['msg'][:45]}")
        print(f"Commits: {len(session)}")
        print(f"Span: {span}")
        print(f"Estimated active: {active}")

        if len(session) <= 5:
            print("Commits:")
            for c in session:
                print(f"  {format_local_time(c)} - {c['msg'][:55]}")

    # Analyze Claude sessions
    print("\n" + "=" * 80)
    print("CLAUDE (AI-ASSISTED) SESSIONS")
    print("=" * 80)
    print("Note: These represent AI-generated commits. You were actively")
    print("prompting/reviewing, but not writing this code directly.")

    total_claude_time = timedelta()

    for i, session in enumerate(claude_sessions, 1):
        first = session[0]
        last = session[-1]
        span = last['ts_utc'] - first['ts_utc']
        active = estimate_session_time(session)
        total_claude_time += active

        print(f"\n--- Claude Session {i} ---")
        print(f"Commits: {len(session)}, Span: {span}")
        for c in session:
            print(f"  {c['ts_utc'].strftime('%H:%M')} UTC - {c['msg'][:55]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    human_hours = total_human_time.total_seconds() / 3600
    claude_hours = total_claude_time.total_seconds() / 3600

    print(f"\nYour direct coding time: {total_human_time} ({human_hours:.1f} hours)")
    print(f"AI-assisted session time: {total_claude_time} ({claude_hours:.1f} hours)")
    print(f"\nTotal active time: {total_human_time + total_claude_time}")
    print(f"                 = {human_hours + claude_hours:.1f} hours")

    return total_human_time, total_claude_time, sessions_by_author


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    commits_file = script_dir / 'commits.txt'
    analyze_commits(commits_file)

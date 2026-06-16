"""Generate IPL 2026 Schedule — 70 matches, exactly 14 per team."""
import pandas as pd
import random
from pathlib import Path
from itertools import combinations

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "2026_schedule.csv"

TEAMS = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

VENUES = {
    "Chennai Super Kings": ("MA Chidambaram Stadium, Chennai", "Chennai"),
    "Delhi Capitals": ("Arun Jaitley Stadium, Delhi", "Delhi"),
    "Gujarat Titans": ("Narendra Modi Stadium, Ahmedabad", "Ahmedabad"),
    "Kolkata Knight Riders": ("Eden Gardens, Kolkata", "Kolkata"),
    "Lucknow Super Giants": ("BRSABV Ekana Cricket Stadium, Lucknow", "Lucknow"),
    "Mumbai Indians": ("Wankhede Stadium, Mumbai", "Mumbai"),
    "Punjab Kings": ("PCA Stadium, Mohali", "Mohali"),
    "Rajasthan Royals": ("Sawai Mansingh Stadium, Jaipur", "Jaipur"),
    "Royal Challengers Bengaluru": ("M Chinnaswamy Stadium, Bengaluru", "Bengaluru"),
    "Sunrisers Hyderabad": ("Rajiv Gandhi Stadium, Hyderabad", "Hyderabad"),
}


def find_5_regular_subgraph():
    """Find 25 pairs from K_10 such that each vertex has degree 5."""
    # K_10 has 45 edges. Need 25 edges forming a 5-regular graph.
    # Use 1-factorization: K_10 decomposes into 9 perfect matchings of 5 edges each.
    # Pick any 5 matchings = 25 edges, each vertex degree 5.
    n = 10

    # Round-robin 1-factorization
    vertices = list(range(n))
    matchings = []

    for r in range(n - 1):
        matching = []
        # Fix vertex 0, pair it with vertex r
        # Pair remaining: (1+r, n-1), (2+r, n-2), etc (mod n-1)
        paired = {0, r}
        matching.append((0, r))
        for i in range(1, n // 2):
            a = (r + i) % (n - 1)
            if a == 0:
                a = n - 1
            b = (r + n - 1 - i) % (n - 1)
            if b == 0:
                b = n - 1
            # Fix: use proper 1-factorization
            pass

    # Simpler: use known 1-factorization for K_10
    # Place vertices 1..9 in a circle, vertex 0 in center
    # Round r: 0 vs r, then pair (r+i) vs (r-i) mod 9 for i=1..4
    matchings = []
    for r in range(1, 10):
        m = [(0, r)]
        used = {0, r}
        for i in range(1, 5):
            a = ((r - 1 + i) % 9) + 1
            b = ((r - 1 - i) % 9) + 1
            if a not in used and b not in used and a != b:
                m.append((min(a, b), max(a, b)))
                used.add(a)
                used.add(b)
        if len(m) == 5:
            matchings.append(m)

    # If we got 9 matchings, pick 5
    if len(matchings) >= 5:
        random.seed(42)
        selected = random.sample(matchings, 5)
    else:
        # Fallback: explicit construction
        # 5 matchings of 5 edges each, covering all 10 vertices
        selected = [
            [(0,1),(2,3),(4,5),(6,7),(8,9)],
            [(0,2),(1,4),(3,6),(5,8),(7,9)],
            [(0,3),(1,5),(2,7),(4,9),(6,8)],
            [(0,4),(1,6),(2,8),(3,9),(5,7)],
            [(0,5),(1,7),(2,9),(3,8),(4,6)],
        ]

    extras = [e for m in selected for e in m]
    return extras


def generate_schedule():
    random.seed(42)

    pairs = list(combinations(range(10), 2))  # 45 base
    extras = find_5_regular_subgraph()  # 25 extras, each team degree 5

    all_pairings = pairs + extras  # 70

    # Verify
    team_count = {}
    for p in all_pairings:
        team_count[p[0]] = team_count.get(p[0], 0) + 1
        team_count[p[1]] = team_count.get(p[1], 0) + 1
    for t in range(10):
        assert team_count[t] == 14, f"Team {TEAMS[t]}: {team_count[t]} matches"

    # Balance home/away
    team_home = {i: 0 for i in range(10)}
    random.shuffle(all_pairings)

    home_away = []
    for p in all_pairings:
        if team_home[p[0]] <= team_home[p[1]]:
            home_away.append((p[0], p[1]))
            team_home[p[0]] += 1
        else:
            home_away.append((p[1], p[0]))
            team_home[p[1]] += 1

    # Build schedule
    schedule = []
    mid = 1
    date = pd.Timestamp("2026-03-21")

    for home, away in home_away:
        venue, city = VENUES[TEAMS[home]]
        toss = TEAMS[random.choice([home, away])]
        decision = "field" if random.random() < 0.65 else "bat"

        schedule.append({
            "match_id": mid, "date": date.strftime("%Y-%m-%d"),
            "team1": TEAMS[home], "team2": TEAMS[away],
            "venue": venue, "city": city,
            "toss_winner": toss, "toss_decision": decision,
        })
        mid += 1
        if mid % 2 == 0:
            date += pd.Timedelta(days=1)

    return pd.DataFrame(schedule).sort_values("match_id").reset_index(drop=True)


def main():
    schedule = generate_schedule()
    print(f"Matches: {len(schedule)}")
    print(f"Dates: {schedule['date'].min()} to {schedule['date'].max()}")
    print("\nMatches per team:")
    for t in TEAMS:
        n = len(schedule[(schedule["team1"] == t) | (schedule["team2"] == t)])
        home = len(schedule[schedule["team1"] == t])
        away = n - home
        print(f"  {t:<35} {n:2d} (H:{home} A:{away})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.titleweight"] = "bold"

# Brand Colors
MI_BLUE = "#004BA0"
MI_GOLD = "#D1AB3E"
MI_PALETTE = [MI_BLUE, "#A9A9A9"] 

# --- Data Loading & Prep ---
print("--- MI Champion DNA Analytics ---")
if not os.path.exists("IPL.csv"):
    print("❌ Error: IPL.csv not found.")
    exit()

print("Loading Data...")
df = pd.read_csv("IPL.csv", low_memory=False)

# Helper: Clean Season Year
def clean_season(s):
    s = str(s)
    if "/" in s:
        return int(s.split("/")[0])
    try:
        return int(s)
    except ValueError:
        return 0

df["season_year"] = df["season"].apply(clean_season)

# Define Champion Years
CHAMPION_YEARS = [2013, 2015, 2017, 2019, 2020]
df["is_champion_year"] = df["season_year"].isin(CHAMPION_YEARS)
df["period"] = df["is_champion_year"].map({True: "Champion Years", False: "Other Years"})

# Pre-calculate
df["is_wicket"] = df["wicket_kind"].notna().astype(int)
df["is_dot"] = (df["runs_total"] == 0).astype(int)

# ==========================================
# 1. Statistical Baseline (Batting & Bowling)
# ==========================================
print("\n[1/4] Generating Core Metrics Analysis...")

# Batting
mi_bat = df[df["batting_team"] == "Mumbai Indians"].copy()
bat_metrics = mi_bat.groupby("period").agg(
    runs=("runs_batter", "sum"),
    wickets=("is_wicket", "sum"),
    balls=("valid_ball", "sum")
).reset_index()
bat_metrics["avg"] = bat_metrics["runs"] / bat_metrics["wickets"]
bat_metrics["sr"] = (bat_metrics["runs"] / bat_metrics["balls"]) * 100

# Bowling
mi_bowling = df[df["bowling_team"] == "Mumbai Indians"].copy()
bowl_metrics = mi_bowling.groupby("period").agg(
    runs_conceded=("runs_total", "sum"),
    wickets=("bowler_wicket", "sum"),
    balls=("valid_ball", "sum")
).reset_index()
bowl_metrics["avg"] = bowl_metrics["runs_conceded"] / bowl_metrics["wickets"]
bowl_metrics["econ"] = (bowl_metrics["runs_conceded"] / bowl_metrics["balls"]) * 6

# Plot 1
fig1, ax = plt.subplots(2, 2, figsize=(16, 10))
fig1.suptitle("Core Metrics: Champions vs Others", fontsize=24)

sns.barplot(data=bat_metrics, x="period", y="avg", ax=ax[0,0], palette=MI_PALETTE)
ax[0,0].set_title("Batting Avg")
ax[0,0].bar_label(ax[0,0].containers[0], fmt="%.2f")

sns.barplot(data=bat_metrics, x="period", y="sr", ax=ax[0,1], palette=MI_PALETTE)
ax[0,1].set_title("Batting Strike Rate")
ax[0,1].bar_label(ax[0,1].containers[0], fmt="%.1f")

sns.barplot(data=bowl_metrics, x="period", y="avg", ax=ax[1,0], palette=MI_PALETTE)
ax[1,0].set_title("Bowling Avg (Lower is Better)")
ax[1,0].bar_label(ax[1,0].containers[0], fmt="%.2f")

sns.barplot(data=bowl_metrics, x="period", y="econ", ax=ax[1,1], palette=MI_PALETTE)
ax[1,1].set_title("Bowling Economy (Lower is Better)")
ax[1,1].bar_label(ax[1,1].containers[0], fmt="%.2f")

plt.tight_layout()
plt.show(block=False) # Show non-blocking so script continues, or rely on end block

# ==========================================
# 2. Phase Analysis
# ==========================================
print("[2/4] Generating Phase Analysis...")
def get_phase(over):
    if over < 6: return "Powerplay"
    elif over >= 15: return "Death"
    else: return "Middle"

mi_bat["phase"] = mi_bat["over"].apply(get_phase)
mi_bowling["phase"] = mi_bowling["over"].apply(get_phase)

pp_bat = mi_bat[mi_bat["phase"] == "Powerplay"].groupby("period").agg(
    runs=("runs_total", "sum"),
    overs=("valid_ball", lambda x: x.count()/6)
).reset_index()
pp_bat["rpo"] = pp_bat["runs"] / pp_bat["overs"]

death_bowl = mi_bowling[mi_bowling["phase"] == "Death"].groupby("period").agg(
    runs=("runs_total", "sum"),
    overs=("valid_ball", lambda x: x.count()/6)
).reset_index()
death_bowl["econ"] = death_bowl["runs"] / death_bowl["overs"]

# Plot 2
fig2, ax = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Phase Dominance", fontsize=22)

sns.barplot(data=pp_bat, x="period", y="rpo", ax=ax[0], palette=MI_PALETTE)
ax[0].set_title("Powerplay Batting (Runs per Over)")
ax[0].bar_label(ax[0].containers[0], fmt="%.2f")

sns.barplot(data=death_bowl, x="period", y="econ", ax=ax[1], palette=MI_PALETTE)
ax[1].set_title("Death Bowling Economy (Lower is Better)")
ax[1].bar_label(ax[1].containers[0], fmt="%.2f")

plt.tight_layout()
plt.show(block=False)

# ==========================================
# 3. Strategy & Fortress
# ==========================================
print("[3/4] Generating Strategy & Fortress Analysis...")
match_data = df[["match_id", "season_year", "venue", "toss_winner", "toss_decision", "match_won_by"]].drop_duplicates()
match_data["is_champion_year"] = match_data["season_year"].isin(CHAMPION_YEARS)
match_data["period"] = match_data["is_champion_year"].map({True: "Champion Years", False: "Other Years"})
match_data["mi_win"] = (match_data["match_won_by"] == "Mumbai Indians").astype(int)

mi_innings = df[(df["batting_team"] == "Mumbai Indians")].groupby(["match_id"])["innings"].min().reset_index()
match_data = pd.merge(match_data, mi_innings, on="match_id", how="left")
match_data["strategy"] = match_data["innings"].map({1: "Defending", 2: "Chasing"})
mi_matches_strat = match_data.dropna(subset=["strategy"])

strat_stats = mi_matches_strat.groupby(["period", "strategy"]).agg(
    played=("match_id", "count"),
    won=("mi_win", "sum")
).reset_index()
strat_stats["win_pct"] = (strat_stats["won"] / strat_stats["played"]) * 100

# Wankhede Stats
wankhede = match_data[
    (match_data["venue"].str.contains("Wankhede", case=False, na=False)) & 
    ((df["batting_team"] == "Mumbai Indians") | (df["bowling_team"] == "Mumbai Indians"))
].drop_duplicates(subset=["match_id"])
# Note: Above filter for MI involvement is technically loose because we filtered DF earlier, 
# but strictly speaking match_data is from full DF. 
# Let's trust the logic that we only care about if MI played.
# Actually, let's re-verify MI played in these matches.
# 'match_data' currently contains ALL matches.
# Need to filter match_data for matches where MI played.

mi_involved_ids = df[(df["batting_team"] == "Mumbai Indians") | (df["bowling_team"] == "Mumbai Indians")]["match_id"].unique()
wankhede = wankhede[wankhede["match_id"].isin(mi_involved_ids)]

home_stats = wankhede.groupby("period").agg(
    played=("match_id", "count"),
    won=("mi_win", "sum")
).reset_index()
home_stats["win_pct"] = (home_stats["won"] / home_stats["played"]) * 100


fig3, ax = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("Strategy & Home Advantage", fontsize=22)

sns.barplot(data=strat_stats, x="period", y="win_pct", hue="strategy", ax=ax[0], palette="viridis")
ax[0].set_title("Win %: Chasing vs Defending")
ax[0].set_ylim(0, 100)

sns.barplot(data=home_stats, x="period", y="win_pct", ax=ax[1], palette=MI_PALETTE)
ax[1].set_title("Fortress Wankhede Win %")
ax[1].bar_label(ax[1].containers[0], fmt="%.1f%%")
ax[1].set_ylim(0, 100)

plt.tight_layout()
plt.show(block=False)

# ==========================================
# 4. Heroes
# ==========================================
print("[4/4] Identifying Champion Heroes...")
champ_bat = mi_bat[mi_bat["is_champion_year"] == True]
champ_bowl = mi_bowling[mi_bowling["is_champion_year"] == True]

top_batters = champ_bat.groupby("batter")["runs_batter"].sum().nlargest(5).reset_index()
top_bowlers = champ_bowl.groupby("bowler")["bowler_wicket"].sum().nlargest(5).reset_index()

fig4, ax = plt.subplots(1, 2, figsize=(18, 7))
fig4.suptitle("Heroes of the 5 Titles", fontsize=24)

sns.barplot(data=top_batters, x="runs_batter", y="batter", ax=ax[0], palette="Blues_r")
ax[0].set_title("Most Runs")
ax[0].bar_label(ax[0].containers[0])

sns.barplot(data=top_bowlers, x="bowler_wicket", y="bowler", ax=ax[1], palette="Oranges_r")
ax[1].set_title("Most Wickets")
ax[1].bar_label(ax[1].containers[0])

plt.tight_layout()
print("\n✅ Analysis Complete. Displaying charts...")
plt.show() # Block here to keep windows open

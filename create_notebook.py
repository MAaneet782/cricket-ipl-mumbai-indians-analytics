
import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🏆 Mumbai Indians: The DNA of Champions (Detailed Analysis)\n",
    "\n",
    "## 📊 Winning vs. Non-Winning Years: The Comprehensive Comparison\n",
    "This notebook performs a deep-dive analysis to understand exactly **what changed** when Mumbai Indians won the trophy versus when they didn't. We compare the **5 Champion Seasons (2013, 2015, 2017, 2019, 2020)** against the non-winning years.\n",
    "\n",
    "### 🔍 Analysis Modules:\n",
    "1.  **Core Metrics**: Batting & Bowling Averages.\n",
    "2.  **Phase Dominance**: Powerplay & Death Overs comparison.\n",
    "3.  **Strategic Strength**: Chasing vs. Defending records.\n",
    "4.  **Fortress Wankhede**: Home ground performance.\n",
    "5.  **The Toss Factor**: Luck vs. Skill.\n",
    "6.  **Champion Heroes**: The players who stepped up."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# --- Configuration ---\n",
    "sns.set_theme(style=\"whitegrid\", context=\"notebook\")\n",
    "plt.rcParams[\"figure.figsize\"] = (14, 7)\n",
    "plt.rcParams[\"axes.titlesize\"] = 16\n",
    "plt.rcParams[\"axes.titleweight\"] = \"bold\"\n",
    "\n",
    "# Brand Colors\n",
    "MI_BLUE = \"#004BA0\"\n",
    "MI_GOLD = \"#D1AB3E\"\n",
    "MI_PALETTE = [MI_BLUE, \"#A9A9A9\"] # Blue for Champions, Gray for Others\n",
    "\n",
    "print(\"Libraries Loaded & Theme Set.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Data Loading & Prep ---\n",
    "try:\n",
    "    df = pd.read_csv(\"IPL.csv\", low_memory=False)\n",
    "    print(f\"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
    "except FileNotFoundError:\n",
    "    print(\"❌ Error: IPL.csv not found. Please ensure the data file is in the directory.\")\n",
    "\n",
    "# Helper: Clean Season Year\n",
    "def clean_season(s):\n",
    "    s = str(s)\n",
    "    if \"/\" in s:\n",
    "        return int(s.split(\"/\")[0])\n",
    "    try:\n",
    "        return int(s)\n",
    "    except ValueError:\n",
    "        return 0\n",
    "\n",
    "df[\"season_year\"] = df[\"season\"].apply(clean_season)\n",
    "\n",
    "# Define Champion Years\n",
    "CHAMPION_YEARS = [2013, 2015, 2017, 2019, 2020]\n",
    "df[\"is_champion_year\"] = df[\"season_year\"].isin(CHAMPION_YEARS)\n",
    "df[\"period\"] = df[\"is_champion_year\"].map({True: \"🏆 Champion Years\", False: \"Other Years\"})\n",
    "\n",
    "# Pre-calculate some columns\n",
    "df[\"is_wicket\"] = df[\"wicket_kind\"].notna().astype(int)\n",
    "df[\"is_dot\"] = (df[\"runs_total\"] == 0).astype(int)\n",
    "df[\"is_four\"] = (df[\"runs_batter\"] == 4).astype(int)\n",
    "df[\"is_six\"] = (df[\"runs_batter\"] == 6).astype(int)\n",
    "\n",
    "print(\"Data Preparation Complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 📈 The Statistical Baseline\n",
    "Did MI score more runs per wicket or concede fewer runs per wicket in winning years?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Batting Overall ---\n",
    "mi_bat = df[df[\"batting_team\"] == \"Mumbai Indians\"].copy()\n",
    "bat_metrics = mi_bat.groupby(\"period\").agg(\n",
    "    runs=(\"runs_batter\", \"sum\"),\n",
    "    wickets=(\"is_wicket\", \"sum\"),\n",
    "    balls=(\"valid_ball\", \"sum\")\n",
    ").reset_index()\n",
    "bat_metrics[\"avg\"] = bat_metrics[\"runs\"] / bat_metrics[\"wickets\"]\n",
    "bat_metrics[\"sr\"] = (bat_metrics[\"runs\"] / bat_metrics[\"balls\"]) * 100\n",
    "\n",
    "# --- Bowling Overall ---\n",
    "mi_bowling = df[df[\"bowling_team\"] == \"Mumbai Indians\"].copy()\n",
    "bowl_metrics = mi_bowling.groupby(\"period\").agg(\n",
    "    runs_conceded=(\"runs_total\", \"sum\"),\n",
    "    wickets=(\"bowler_wicket\", \"sum\"),\n",
    "    balls=(\"valid_ball\", \"sum\")\n",
    ").reset_index()\n",
    "bowl_metrics[\"avg\"] = bowl_metrics[\"runs_conceded\"] / bowl_metrics[\"wickets\"]\n",
    "bowl_metrics[\"econ\"] = (bowl_metrics[\"runs_conceded\"] / bowl_metrics[\"balls\"]) * 6\n",
    "\n",
    "# --- Visualization ---\n",
    "fig, ax = plt.subplots(1, 4, figsize=(20, 5))\n",
    "\n",
    "sns.barplot(data=bat_metrics, x=\"period\", y=\"avg\", ax=ax[0], palette=MI_PALETTE)\n",
    "ax[0].set_title(\"Batting Avg (Higher=Better)\")\n",
    "ax[0].bar_label(ax[0].containers[0], fmt=\"%.2f\")\n",
    "\n",
    "sns.barplot(data=bat_metrics, x=\"period\", y=\"sr\", ax=ax[1], palette=MI_PALETTE)\n",
    "ax[1].set_title(\"Batting Strike Rate\")\n",
    "ax[1].bar_label(ax[1].containers[0], fmt=\"%.1f\")\n",
    "\n",
    "sns.barplot(data=bowl_metrics, x=\"period\", y=\"avg\", ax=ax[2], palette=MI_PALETTE)\n",
    "ax[2].set_title(\"Bowling Avg (Lower=Better)\")\n",
    "ax[2].bar_label(ax[2].containers[0], fmt=\"%.2f\")\n",
    "\n",
    "sns.barplot(data=bowl_metrics, x=\"period\", y=\"econ\", ax=ax[3], palette=MI_PALETTE)\n",
    "ax[3].set_title(\"Bowling Economy (Lower=Better)\")\n",
    "ax[3].bar_label(ax[3].containers[0], fmt=\"%.2f\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. ⚡ Phase Analysis: Powerplay & Death\n",
    "Champions are often defined by how they start (Powerplay) and how they finish (Death Overs). This section separates the game into phases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_phase(over):\n",
    "    if over < 6: return \"Powerplay\"\n",
    "    elif over >= 15: return \"Death\"\n",
    "    else: return \"Middle\"\n",
    "\n",
    "mi_bat[\"phase\"] = mi_bat[\"over\"].apply(get_phase)\n",
    "mi_bowling[\"phase\"] = mi_bowling[\"over\"].apply(get_phase)\n",
    "\n",
    "# --- Batting: Powerplay Runs per Over ---\n",
    "pp_bat = mi_bat[mi_bat[\"phase\"] == \"Powerplay\"].groupby(\"period\").agg(\n",
    "    runs=(\"runs_total\", \"sum\"),\n",
    "    overs=(\"valid_ball\", lambda x: x.count()/6)\n",
    ").reset_index()\n",
    "pp_bat[\"rpo\"] = pp_bat[\"runs\"] / pp_bat[\"overs\"]\n",
    "\n",
    "# --- Bowling: Death Overs Economy ---\n",
    "death_bowl = mi_bowling[mi_bowling[\"phase\"] == \"Death\"].groupby(\"period\").agg(\n",
    "    runs=(\"runs_total\", \"sum\"),\n",
    "    overs=(\"valid_ball\", lambda x: x.count()/6)\n",
    ").reset_index()\n",
    "death_bowl[\"econ\"] = death_bowl[\"runs\"] / death_bowl[\"overs\"]\n",
    "\n",
    "# --- Visualize ---\n",
    "fig, ax = plt.subplots(1, 2, figsize=(16, 6))\n",
    "\n",
    "sns.barplot(data=pp_bat, x=\"period\", y=\"rpo\", ax=ax[0], palette=MI_PALETTE)\n",
    "ax[0].set_title(\"Batting: Powerplay Run Rate\")\n",
    "ax[0].bar_label(ax[0].containers[0], fmt=\"%.2f\")\n",
    "\n",
    "sns.barplot(data=death_bowl, x=\"period\", y=\"econ\", ax=ax[1], palette=MI_PALETTE)\n",
    "ax[1].set_title(\"Bowling: Death Over Economy (Crucial!)\")\n",
    "ax[1].bar_label(ax[1].containers[0], fmt=\"%.2f\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. ⚔️ Strategy: Chasing vs. Defending\n",
    "Did MI become a better chasing team or a defending fortress in their winning years?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Match Level Data\n",
    "match_data = df[[\"match_id\", \"season_year\", \"toss_winner\", \"toss_decision\", \"match_won_by\"]].drop_duplicates()\n",
    "match_data[\"is_champion_year\"] = match_data[\"season_year\"].isin(CHAMPION_YEARS)\n",
    "match_data[\"period\"] = match_data[\"is_champion_year\"].map({True: \"🏆 Champion Years\", False: \"Other Years\"})\n",
    "\n",
    "# Add MI Win Flag\n",
    "match_data[\"mi_win\"] = (match_data[\"match_won_by\"] == \"Mumbai Indians\").astype(int)\n",
    "\n",
    "# Determine if MI Bat First or Second\n",
    "# Note: We need to merge with inning data to be 100% sure, but using toss decision is a good proxy if correct.\n",
    "# Better approach: Check if MI batted in inning 1 or 2 from original DF.\n",
    "\n",
    "mi_innings = df[(df[\"batting_team\"] == \"Mumbai Indians\")].groupby([\"match_id\"])[\"innings\"].min().reset_index()\n",
    "match_data = pd.merge(match_data, mi_innings, on=\"match_id\", how=\"left\")\n",
    "\n",
    "# inning 1 = Defending (Setting Target), inning 2 = Chasing\n",
    "match_data[\"strategy\"] = match_data[\"innings\"].map({1: \"Defending\", 2: \"Chasing\"})\n",
    "\n",
    "# Filter only MI matches (where strategy is known)\n",
    "mi_matches_strat = match_data.dropna(subset=[\"strategy\"])\n",
    "\n",
    "strat_stats = mi_matches_strat.groupby([\"period\", \"strategy\"]).agg(\n",
    "    played=(\"match_id\", \"count\"),\n",
    "    won=(\"mi_win\", \"sum\")\n",
    ").reset_index()\n",
    "\n",
    "strat_stats[\"win_pct\"] = (strat_stats[\"won\"] / strat_stats[\"played\"]) * 100\n",
    "\n",
    "# Visualize\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(data=strat_stats, x=\"period\", y=\"win_pct\", hue=\"strategy\", palette=\"viridis\")\n",
    "plt.title(\"Win % by Strategy: Chasing vs Defending\", fontsize=18)\n",
    "plt.ylabel(\"Win %\")\n",
    "plt.legend(title=\"Strategy\")\n",
    "plt.ylim(0, 100)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 🏟️ Fortress Wankhede\n",
    "Home advantage is key in IPL. How much did Wankhede contribute?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "match_cols = [\"match_id\", \"season_year\", \"venue\", \"match_won_by\"]\n",
    "wankhede = df[\n",
    "    (df[\"venue\"].str.contains(\"Wankhede\", case=False, na=False)) & \n",
    "    ((df[\"batting_team\"] == \"Mumbai Indians\") | (df[\"bowling_team\"] == \"Mumbai Indians\"))\n",
    "][match_cols].drop_duplicates()\n",
    "\n",
    "wankhede[\"is_champion_year\"] = wankhede[\"season_year\"].isin(CHAMPION_YEARS)\n",
    "wankhede[\"period\"] = wankhede[\"is_champion_year\"].map({True: \"🏆 Champion Years\", False: \"Other Years\"})\n",
    "wankhede[\"mi_win\"] = (wankhede[\"match_won_by\"] == \"Mumbai Indians\").astype(int)\n",
    "\n",
    "home_stats = wankhede.groupby(\"period\").agg(\n",
    "    played=(\"match_id\", \"count\"),\n",
    "    won=(\"mi_win\", \"sum\")\n",
    ").reset_index()\n",
    "home_stats[\"win_pct\"] = (home_stats[\"won\"] / home_stats[\"played\"]) * 100\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "ax = sns.barplot(data=home_stats, x=\"period\", y=\"win_pct\", palette=MI_PALETTE)\n",
    "ax.set_title(\"Win % at Wankhede Stadium\")\n",
    "ax.bar_label(ax.containers[0], fmt=\"%.1f%%\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. ⭐ Champion Heroes\n",
    "Who were the MVPs of the 5 Titles? (2013, 2015, 2017, 2019, 2020)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "champ_bat = mi_bat[mi_bat[\"is_champion_year\"] == True]\n",
    "champ_bowl = mi_bowling[mi_bowling[\"is_champion_year\"] == True]\n",
    "\n",
    "top_batters = champ_bat.groupby(\"batter\")[\"runs_batter\"].sum().nlargest(5).reset_index()\n",
    "top_bowlers = champ_bowl.groupby(\"bowler\")[\"bowler_wicket\"].sum().nlargest(5).reset_index()\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(18, 7))\n",
    "\n",
    "sns.barplot(data=top_batters, x=\"runs_batter\", y=\"batter\", ax=ax[0], palette=\"Blues_r\")\n",
    "ax[0].set_title(\"Most Runs in 5 Winning Seasons\")\n",
    "ax[0].bar_label(ax[0].containers[0])\n",
    "\n",
    "sns.barplot(data=top_bowlers, x=\"bowler_wicket\", y=\"bowler\", ax=ax[1], palette=\"Oranges_r\")\n",
    "ax[1].set_title(\"Most Wickets in 5 Winning Seasons\")\n",
    "ax[1].bar_label(ax[1].containers[0])\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("mi_champion_analysis.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook recreated successfully!")

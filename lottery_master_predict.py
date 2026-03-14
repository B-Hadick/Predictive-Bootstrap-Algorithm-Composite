import pandas as pd, numpy as np, urllib.request
from io import StringIO
from itertools import combinations
from collections import Counter
import sys, datetime

# ----------------------------------------------------------
def fetch_data(url):
    """Download CSV and normalize column names."""
    csv_data = urllib.request.urlopen(url).read().decode("utf-8")
    df = pd.read_csv(StringIO(csv_data))
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    return df

# ----------------------------------------------------------
def get_column(df, *possible):
    for name in possible:
        key = name.lower().replace(" ", "")
        for c in df.columns:
            if key in c:
                return c
    return None

# ----------------------------------------------------------
def bootstrap_predict(df, white_range, bonus_range, game_label, decay=0.995, smooth=1):
    date_col   = get_column(df, "drawdate", "drawingdate")
    numbers_col= get_column(df, "winningnumbers", "numbers")
    bonus_col  = get_column(df, "powerball", "megaball", "luckyball", "bonus")

    if not date_col or not numbers_col:
        print("DEBUG:", df.columns.tolist())
        raise KeyError(f"⚠️ Missing required columns for {game_label}.")

    df = df.dropna(subset=[date_col, numbers_col])
    df = df.sort_values(date_col)

    weights = decay ** np.arange(len(df))[::-1]
    white_counts = np.zeros(white_range)
    bonus_counts = np.zeros(bonus_range)
    pair_counter = Counter()

    for i, row in enumerate(df[numbers_col]):
        try:
            nums = [int(x) for x in str(row).split()]
        except Exception:
            continue

        # Handle datasets that include bonus number in same field
        if len(nums) > 5:
            whites, bonus_val = nums[:5], nums[5]
        else:
            whites, bonus_val = nums, int(df.iloc[i][bonus_col])

        w = weights[i]
        for n in whites:
            if 1 <= n <= white_range:
                white_counts[n-1] += w
        if 1 <= bonus_val <= bonus_range:
            bonus_counts[bonus_val-1] += w
        for a,b in combinations(sorted(whites),2):
            pair_counter[(a,b)] += w

    white_prob = (white_counts + smooth) / (white_counts.sum() + smooth*white_range)
    bonus_prob = (bonus_counts + smooth) / (bonus_counts.sum() + smooth*bonus_range)

    def draw_one():
        whites = np.sort(np.random.choice(np.arange(1,white_range+1),5,replace=False,p=white_prob))
        bonus = np.random.choice(np.arange(1,bonus_range+1),1,p=bonus_prob)[0]
        freq_score = white_prob[whites-1].sum() + bonus_prob[bonus-1]
        pair_score = sum(pair_counter.get(tuple(sorted(p)),0) for p in combinations(whites,2))
        return (tuple(whites)+(int(bonus),), freq_score + 0.0001*pair_score)

    print(f"\n🔄 Bootstrapping 40000 {game_label} draws ...")
    samples = Counter()
    for _ in range(40000):
        combo, score = draw_one()
        samples[combo] += score

    top5 = samples.most_common(5)
    print(f"\n🎯 Top 5 {game_label} Predictions (stability ranked):")
    for i,(combo,score) in enumerate(top5,1):
        print(f"{i}. {list(combo)} (stability score={score:.2f})")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{game_label.lower().replace(' ','_')}_top5_{timestamp}.csv"
    pd.DataFrame([list(c) for c,_ in top5]).to_csv(filename, index=False)
    print(f"\n💾 Results saved → {filename}\n")

# ----------------------------------------------------------
def run_powerball():
    print("📥 Downloading Powerball data ...")
    df = fetch_data("https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD")
    # Parse 6-number format automatically
    bootstrap_predict(df, white_range=69, bonus_range=26, game_label="Powerball")

def run_megamillions():
    print("📥 Downloading Mega Millions data ...")
    df = fetch_data("https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD")
    bootstrap_predict(df, white_range=70, bonus_range=25, game_label="Mega Millions")

def run_luckyforlife():
    print("📥 Downloading Lucky for Life data ...")
    df = fetch_data("https://data.ny.gov/api/views/h6w8-42p9/rows.csv?accessType=DOWNLOAD")
    bootstrap_predict(df, white_range=48, bonus_range=18, game_label="Lucky for Life")

# ----------------------------------------------------------
def main():
    while True:
        print("""
============================================
🎯 Lottery Predictor Master Script (Final)
============================================
1. Powerball
2. Mega Millions
3. Lucky for Life
0. Exit
============================================
""")
        choice = input("Choose a game: ").strip()
        if choice == "1":
            try: run_powerball()
            except Exception as e: print("❌", e)
        elif choice == "2":
            try: run_megamillions()
            except Exception as e: print("❌", e)
        elif choice == "3":
            try: run_luckyforlife()
            except Exception as e: print("❌", e)
        elif choice == "0":
            sys.exit()
        else:
            print("Please choose 1, 2, 3, or 0.")

if __name__ == "__main__":
    main()
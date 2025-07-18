
#%%
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', None)

def load_data():
    df = pd.read_csv(r"\\192.168.2.19\ai_team\AI Program\Outputs\CompiledPiMachine\CompiledPIMachine.csv", encoding='latin1')

    df = df[~df["MODEL CODE"].isin(['60CAT0203M'])]  # REMOVE SPECIFIC MODEL
    df['S/N'] = df['S/N'].astype(str)
    df = df[df['S/N'].str.len() >= 8]                # REMOVE SERIAL NUMBERS LESS THAN 6 DIGITS
    df = df[~df['MODEL CODE'].str.contains("M")]     # REMOVE MODEL CODES WITH 'M'

    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    today = pd.to_datetime(datetime.now().date())

    # REMOVE RUNS BELOW 10 PER DAY EXCEPT TODAY
    count_df = df.groupby(['MODEL CODE', 'DATE']).size().reset_index(name='COUNT')
    valid = count_df[(count_df['COUNT'] >= 10) | (count_df['DATE'] == today)]
    df = df.merge(valid[['MODEL CODE', 'DATE']], on=['MODEL CODE', 'DATE'], how='inner')

    return df

def build_compiled_frame(df):
    dataList = []
    for a in range(len(df)):
        print(f"ROW: {a}")
        tempdf = df.iloc[[a]]
        row = {
            "DATE": tempdf["DATE"].values[0],
            "TIME": tempdf["TIME"].values[0],
            "MODEL CODE": tempdf["MODEL CODE"].values[0],
            "S/N": tempdf["S/N"].values[0],
            "PASS/NG": tempdf["PASS/NG"].values[0],
            "VOLTAGE MAX (V)": tempdf["VOLTAGE MAX (V)"].values[0],
            "WATTAGE MAX (W)": tempdf["WATTAGE MAX (W)"].values[0],
            "CLOSED PRESSURE_MAX (kPa)": tempdf["CLOSED PRESSURE_MAX (kPa)"].values[0],
            "VOLTAGE Middle (V)": tempdf["VOLTAGE Middle (V)"].values[0],
            "WATTAGE Middle (W)": tempdf["WATTAGE Middle (W)"].values[0],
            "AMPERAGE Middle (A)": tempdf["AMPERAGE Middle (A)"].values[0],
            "CLOSED PRESSURE Middle (kPa)": tempdf["CLOSED PRESSURE Middle (kPa)"].values[0],
            "VOLTAGE MIN (V)": tempdf["VOLTAGE MIN (V)"].values[0],
            "WATTAGE MIN (W)": tempdf["WATTAGE MIN (W)"].values[0],
            "CLOSED PRESSURE MIN (kPa)": tempdf["CLOSED PRESSURE MIN (kPa)"].values[0],
            "V_MAX PASS": None,
            "WATTAGE MAX PASS": None,
            "CLOSED PRESSURE_MAX PASS": None,
            "VOLTAGE Middle PASS": None,
            "WATTAGE Middle (W) PASS": None,
            "AMPERAGE Middle (A) PASS": None,
            "CLOSED PRESSURE Middle (kPa) PASS": None,
            "VOLTAGE MIN (V) PASS": None,
            "WATTAGE MIN (W) PASS": None,
            "CLOSED PRESSURE MIN (kPa) PASS": None,
        }

        if row["PASS/NG"] == 1:
            row["V_MAX PASS"] = row["VOLTAGE MAX (V)"]
            row["WATTAGE MAX PASS"] = row["WATTAGE MAX (W)"]
            row["CLOSED PRESSURE_MAX PASS"] = row["CLOSED PRESSURE_MAX (kPa)"]
            row["VOLTAGE Middle PASS"] = row["VOLTAGE Middle (V)"]
            row["WATTAGE Middle (W) PASS"] = row["WATTAGE Middle (W)"]
            row["AMPERAGE Middle (A) PASS"] = row["AMPERAGE Middle (A)"]
            row["CLOSED PRESSURE Middle (kPa) PASS"] = row["CLOSED PRESSURE Middle (kPa)"]
            row["VOLTAGE MIN (V) PASS"] = row["VOLTAGE MIN (V)"]
            row["WATTAGE MIN (W) PASS"] = row["WATTAGE MIN (W)"]
            row["CLOSED PRESSURE MIN (kPa) PASS"] = row["CLOSED PRESSURE MIN (kPa)"]

        dataList.append(row)

    return pd.DataFrame(dataList)

def compute_model_summary(compiledFrame):
    model_summary = pd.DataFrame()
    today = pd.to_datetime(datetime.now().date())

    for model in compiledFrame["MODEL CODE"].unique():
        modelFrame = compiledFrame[(compiledFrame["MODEL CODE"] == model) & (compiledFrame["DATE"].dt.date < today.date())]
        modelFrame = modelFrame.sort_values("DATE", ascending=False)
        count = 0
        combined_rows = pd.DataFrame()

        for date in modelFrame["DATE"].dt.date.unique():
            temp = modelFrame[modelFrame["DATE"].dt.date == date]
            temp = temp[temp["PASS/NG"] == 1]
            count += len(temp)
            combined_rows = pd.concat([combined_rows, temp])
            if count >= 200:
                break

        if count >= 200:
            ave = combined_rows.mean(numeric_only=True)
            row = {
                "MODEL CODE": model,
                "AVE V_MAX PASS": ave.get("V_MAX PASS", 0),
                "AVE WATTAGE MAX (W)": ave.get("WATTAGE MAX PASS", 0),
                "AVE CLOSED PRESSURE_MAX (kPa)": ave.get("CLOSED PRESSURE_MAX PASS", 0),
                "AVE VOLTAGE Middle (V)": ave.get("VOLTAGE Middle PASS", 0),
                "AVE WATTAGE Middle (W)": ave.get("WATTAGE Middle (W) PASS", 0),
                "AVE AMPERAGE Middle (A)": ave.get("AMPERAGE Middle (A) PASS", 0),
                "AVE CLOSED PRESSURE Middle (kPa)": ave.get("CLOSED PRESSURE Middle (kPa) PASS", 0),
                "AVE VOLTAGE MIN (V)": ave.get("VOLTAGE MIN (V) PASS", 0),
                "AVE WATTAGE MIN (W)": ave.get("WATTAGE MIN (W) PASS", 0),
                "AVE CLOSED PRESSURE MIN (kPa)": ave.get("CLOSED PRESSURE MIN (kPa) PASS", 0)
            }
            model_summary = pd.concat([model_summary, pd.DataFrame([row])], ignore_index=True)

    return model_summary

def add_deviations(compiledFrame, model_summary):
    compiledFrame = compiledFrame.copy()

    for _, summary_row in model_summary.iterrows():
        model = summary_row["MODEL CODE"]
        temp_idx = compiledFrame[compiledFrame["MODEL CODE"] == model].index

        for col in [
            ("V_MAX PASS", "AVE V_MAX PASS"),
            ("WATTAGE MAX PASS", "AVE WATTAGE MAX (W)"),
            ("CLOSED PRESSURE_MAX PASS", "AVE CLOSED PRESSURE_MAX (kPa)"),
            ("VOLTAGE Middle PASS", "AVE VOLTAGE Middle (V)"),
            ("WATTAGE Middle (W) PASS", "AVE WATTAGE Middle (W)"),
            ("AMPERAGE Middle (A) PASS", "AVE AMPERAGE Middle (A)"),
            ("CLOSED PRESSURE Middle (kPa) PASS", "AVE CLOSED PRESSURE Middle (kPa)"),
            ("VOLTAGE MIN (V) PASS", "AVE VOLTAGE MIN (V)"),
            ("WATTAGE MIN (W) PASS", "AVE WATTAGE MIN (W)"),
            ("CLOSED PRESSURE MIN (kPa) PASS", "AVE CLOSED PRESSURE MIN (kPa)")
        ]:
            pass_col, ave_col = col
            ave_val = summary_row[ave_col]
            dev_col = ave_col.replace("AVE", "DEV")
            compiledFrame.loc[temp_idx, ave_col] = ave_val
            compiledFrame.loc[temp_idx, dev_col] = (compiledFrame.loc[temp_idx, pass_col] - ave_val) / ave_val

    return compiledFrame

def main():
    df = load_data()
    compiledFrame = build_compiled_frame(df)
    model_summary = compute_model_summary(compiledFrame)
    compiledFrame = add_deviations(compiledFrame, model_summary)

    print("\n MODEL SUMMARY:\n", model_summary)
    print("\n COMPILED FRAME:\n", compiledFrame)

if __name__ == "__main__":
    main()


# %%

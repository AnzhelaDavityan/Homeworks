import pandas as pd
import numpy as np

def load_smart_furniture_market(file_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(file_path)
    
    df = pd.read_excel(xls, sheet_name=1, skiprows=4, header=None)
    
    df = df.iloc[:, 1:]
    
    df.columns = ["Year", "Smart Furniture Market"]
    
    df = df.dropna().reset_index(drop=True)
    
    df["Year"] = df["Year"].astype(str).str.extract(r'(\d+)').astype(int)
    
    df["Smart Furniture Market"] = df["Smart Furniture Market"].astype(float)
    
    return df.head(10)

def bass_model(t, p, q, M):
    """
    Bass diffusion model equation.
    p: Coefficient of innovation
    q: Coefficient of imitation
    M: Market potential
    t: Time
    """
    N = M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    return N

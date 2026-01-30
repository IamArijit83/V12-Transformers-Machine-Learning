"""
Feature Engineering for Performance Degradation Prediction
Creates variance-based proxies for driver degradation
"""

import pandas as pd
import numpy as np
from scipy import stats

class DegradationFeatureEngineer:
    """
    Create features that proxy for driver performance degradation
    WITHOUT biometric data (using lap variance patterns)
    """
    
    def __init__(self):
        pass
    
    def calculate_rolling_stats(self, df, window=5):
        """
        Calculate rolling statistics for lap times
        Key insight: Variance increases as driver fatigues
        """
        df = df.sort_values(['Driver', 'RaceNumber', 'LapNumber']).copy()
        
        # Rolling mean lap time (per driver per race)
        df['LapTime_RollingMean'] = df.groupby(['Driver', 'RaceNumber'])['LapTime'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std deviation (VARIANCE PROXY)
        df['LapTime_RollingStd'] = df.groupby(['Driver', 'RaceNumber'])['LapTime'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Coefficient of variation (normalized variance)
        df['LapTime_CV'] = df['LapTime_RollingStd'] / df['LapTime_RollingMean']
        
        return df
    
    def calculate_sector_inconsistency(self, df):
        """
        Sector time inconsistency increases with degradation
        """
        df = df.copy()
        
        # Sector time ratios (should be stable)
        df['S1_Ratio'] = df['Sector1Time'] / df['LapTime']
        df['S2_Ratio'] = df['Sector2Time'] / df['LapTime']
        df['S3_Ratio'] = df['Sector3Time'] / df['LapTime']
        
        # Rolling std of sector ratios (inconsistency indicator)
        for sector in ['S1_Ratio', 'S2_Ratio', 'S3_Ratio']:
            df[f'{sector}_Std'] = df.groupby(['Driver', 'RaceNumber'])[sector].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )
        
        # Combined sector inconsistency score
        df['SectorInconsistency_Score'] = (
            df['S1_Ratio_Std'] + df['S2_Ratio_Std'] + df['S3_Ratio_Std']
        ) / 3
        
        return df
    
    def calculate_stint_progression(self, df):
        """
        Tire degradation + driver fatigue within stint
        """
        df = df.copy()
        
        # Lap number within stint
        df['LapInStint'] = df.groupby(['Driver', 'RaceNumber', 'Stint']).cumcount() + 1
        
        # Percentage through stint
        df['StintProgress_Pct'] = df.groupby(['Driver', 'RaceNumber', 'Stint'])['LapInStint'].transform(
            lambda x: x / x.max()
        )
        
        # Lap time degradation rate (compared to stint start)
        df['LapTimeDelta_FromStintStart'] = df.groupby(['Driver', 'RaceNumber', 'Stint'])['LapTime'].transform(
            lambda x: x - x.iloc[0] if len(x) > 0 else 0
        )
        
        return df
    
    def calculate_pace_delta(self, df):
        """
        Delta to fastest lap in race (consistency indicator)
        """
        df = df.copy()
        
        # Fastest lap per race
        df['FastestLap_Race'] = df.groupby('RaceNumber')['LapTime'].transform('min')
        
        # Delta to fastest
        df['DeltaToFastest'] = df['LapTime'] - df['FastestLap_Race']
        
        # Personal best deviation
        df['PersonalBest'] = df.groupby(['Driver', 'RaceNumber'])['LapTime'].transform('min')
        df['DeltaToPersonalBest'] = df['LapTime'] - df['PersonalBest']
        
        return df
    
    def create_degradation_label(self, df, threshold_std=0.5):
        """
        Binary label: Is driver showing degradation signs?
        Based on lap time variance exceeding threshold
        """
        df = df.copy()
        
        # Degradation = high variance + slower than personal best
        df['IsDegraded'] = (
            (df['LapTime_RollingStd'] > threshold_std) & 
            (df['DeltaToPersonalBest'] > 1.0)  # More than 1 sec slower
        ).astype(int)
        
        return df
    
    def engineer_all_features(self, df):
        """
        Apply all feature engineering steps
        """
        print("🔧 Engineering features...")
        
        df = self.calculate_rolling_stats(df)
        df = self.calculate_sector_inconsistency(df)
        df = self.calculate_stint_progression(df)
        df = self.calculate_pace_delta(df)
        df = self.create_degradation_label(df)
        
        # Drop rows with NaN (from rolling windows at start)
        df = df.dropna().reset_index(drop=True)
        
        print(f"✅ Features created. Shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Test feature engineering
    try:
        df = pd.read_csv('data/processed/races_2024_1-3.csv')
        
        engineer = DegradationFeatureEngineer()
        df_features = engineer.engineer_all_features(df)
        
        df_features.to_csv('data/processed/features_engineered.csv', index=False)
        
        print("\n📊 Feature Summary:")
        print(df_features[['Driver', 'LapNumber', 'LapTime', 'LapTime_RollingStd', 
                           'SectorInconsistency_Score', 'IsDegraded']].head(20))
        
        print(f"\nDegradation distribution:")
        print(df_features['IsDegraded'].value_counts())
        
    except FileNotFoundError:
        print("❌ Please run data_loader.py first to generate data!")
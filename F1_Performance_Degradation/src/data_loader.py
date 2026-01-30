"""
F1 Data Loader using FastF1 API
Author: Research Project 2025
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class F1DataLoader:
    """
    Fetch and cache F1 data using FastF1 API
    """
    
    def __init__(self, cache_dir='data/raw'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable FastF1 caching
        fastf1.Cache.enable_cache(str(self.cache_dir))
        print(f"✅ Cache enabled at: {self.cache_dir.absolute()}")
    
    def get_season_schedule(self, year=2024):
        """Get full season schedule"""
        schedule = fastf1.get_event_schedule(year)
        return schedule
    
    def load_race_session(self, year, round_number, session_type='R'):
        """
        Load a specific race session
        
        Args:
            year: Season year (e.g., 2024)
            round_number: Race number (1-24)
            session_type: 'R' (Race), 'Q' (Qualifying), 'FP1', 'FP2', 'FP3', 'S' (Sprint)
        """
        print(f"Loading {year} Round {round_number} - {session_type}...")
        
        session = fastf1.get_session(year, round_number, session_type)
        session.load()
        
        return session
    
    def extract_lap_data(self, session, driver=None):
        """
        Extract lap times and basic features
        
        Returns: DataFrame with lap-by-lap data
        """
        laps = session.laps
        
        if driver:
            laps = laps.pick_driver(driver)
        
        # Select relevant columns
        lap_data = laps[[
            'DriverNumber', 'Driver', 'LapNumber', 'LapTime', 
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Compound', 'TyreLife', 'Stint', 'TrackStatus',
            'IsPersonalBest', 'IsAccurate'
        ]].copy()
        
        # Convert timedelta to seconds
        for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
            lap_data[col] = lap_data[col].dt.total_seconds()
        
        # Remove invalid laps
        lap_data = lap_data[lap_data['IsAccurate'] == True].reset_index(drop=True)
        
        return lap_data
    
    def get_multi_race_data(self, year, races, session_type='R', drivers=None):
        """
        Fetch data from multiple races
        
        Args:
            year: Season
            races: List of race numbers [1, 2, 3, ...]
            session_type: 'R', 'Q', etc.
            drivers: List of driver abbreviations ['VER', 'HAM'] or None for all
        
        Returns: Combined DataFrame
        """
        all_data = []
        
        for race_num in tqdm(races, desc="Fetching races"):
            try:
                session = self.load_race_session(year, race_num, session_type)
                
                if drivers:
                    for driver in drivers:
                        lap_data = self.extract_lap_data(session, driver)
                        lap_data['RaceNumber'] = race_num
                        lap_data['Year'] = year
                        lap_data['EventName'] = session.event['EventName']
                        all_data.append(lap_data)
                else:
                    lap_data = self.extract_lap_data(session)
                    lap_data['RaceNumber'] = race_num
                    lap_data['Year'] = year
                    lap_data['EventName'] = session.event['EventName']
                    all_data.append(lap_data)
                    
            except Exception as e:
                print(f"❌ Error loading Race {race_num}: {e}")
                continue
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def save_data(self, data, filename):
        """Save processed data"""
        save_path = Path('data/processed') / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)
        print(f"💾 Saved: {save_path}")


if __name__ == "__main__":
    # Test the data loader
    loader = F1DataLoader()
    
    # Get 2024 schedule
    schedule = loader.get_season_schedule(2024)
    print("\n2024 F1 Season Schedule:")
    print(schedule[['RoundNumber', 'EventName', 'EventDate']])
    
    # Load single race
    print("\n" + "="*50)
    session = loader.load_race_session(2024, 1, 'R')
    lap_data = loader.extract_lap_data(session)
    print(f"\nBahrain GP 2024 - Total valid laps: {len(lap_data)}")
    print(lap_data.head())
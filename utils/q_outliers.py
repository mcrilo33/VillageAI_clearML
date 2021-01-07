import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def ___duration(df):
    
    def _duration(gb):
        return gb.peak_time.max() - gb.peak_time.min()
    
    return df.groupby(['timestamp']).apply(_duration)

def ___amp(df):
    
    def _amp(gb):
        s =  np.stack(gb.signal)
        return s.min() - s.max()
    
    return df.groupby(
        ['timestamp', 'axle']
    ).apply(_amp).groupby('timestamp').mean()

def ___gradient(df):
    
    def _gradient(gb):
        
        return gb.signal.apply(lambda x: np.max(np.abs(np.gradient(x, axis=0)))).max()
    
    return df.groupby(
        ['timestamp', 'axle']
    ).apply(_gradient).groupby('timestamp').mean()

def ___width(df): 

    return df.set_index(
        ['timestamp','axle']
    )['width'].groupby('timestamp').mean()

def ___left_width(df): 

    return df.set_index(
        ['timestamp','axle']
    )['left_width'].groupby('timestamp').mean()

def ___right_width(df): 

    return df.set_index(
        ['timestamp','axle']
    )['right_width'].groupby('timestamp').mean()

def ___max_pos(df): 

    return df.set_index(
        ['timestamp','axle']
    )['max_pos'].groupby('timestamp').mean()

def ___median(df): 

    return df.set_index(
        ['timestamp','axle']
    )['median'].groupby('timestamp').mean()

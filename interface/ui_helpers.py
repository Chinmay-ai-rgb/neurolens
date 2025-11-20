import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fixation_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if 'left_x' in df.columns and 'left_y' in df.columns:
        valid = ~(df['left_x'].isna() | df['left_y'].isna())
        if valid.sum() > 0:
            ax.scatter(df[valid]['left_x'], df[valid]['left_y'], alpha=0.5, s=10)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Fixation Stability')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    return fig


def plot_saccade_velocity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if 'timestamp' in df.columns and 'left_x' in df.columns:
        df_clean = df[~(df['timestamp'].isna() | df['left_x'].isna())].copy()
        if len(df_clean) > 1:
            df_clean = df_clean.sort_values('timestamp')
            dt = df_clean['timestamp'].diff()
            dx = df_clean['left_x'].diff()
            velocity = np.abs(dx / dt)
            velocity = velocity.fillna(0)
            
            ax.plot(df_clean['timestamp'], velocity, linewidth=1)
            ax.axhline(y=50, color='r', linestyle='--', label='Saccade Threshold')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (px/s)')
    ax.set_title('Eye Movement Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_pursuit_overlay(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'timestamp' in df.columns and 'left_x' in df.columns and 'target_x' in df.columns:
        df_clean = df[~(df['timestamp'].isna() | df['left_x'].isna() | df['target_x'].isna())].copy()
        if len(df_clean) > 0:
            df_clean = df_clean.sort_values('timestamp')
            ax.plot(df_clean['timestamp'], df_clean['target_x'], 'g-', label='Target', linewidth=2)
            ax.plot(df_clean['timestamp'], df_clean['left_x'], 'b-', label='Eye', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (pixels)')
    ax.set_title('Smooth Pursuit Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_anti_saccade_performance(error_rate: float):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    categories = ['Correct', 'Error']
    values = [1 - error_rate, error_rate]
    colors = ['green', 'red']
    
    ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Proportion')
    ax.set_title('Anti-Saccade Performance')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_risk_scores(risks: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conditions = list(risks.keys())
    scores = list(risks.values())
    
    colors = []
    for score in scores:
        if score > 0.7:
            colors.append('red')
        elif score > 0.4:
            colors.append('orange')
        else:
            colors.append('green')
    
    bars = ax.barh(conditions, scores, color=colors, alpha=0.7)
    ax.set_xlabel('Risk Score')
    ax.set_title('Neurological Risk Assessment')
    ax.set_xlim(0, 1)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.02, i, f'{score:.2f}', va='center')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig


def format_risk_indicator(risk: float) -> str:
    if risk > 0.7:
        return "🔴 High"
    elif risk > 0.4:
        return "🟠 Moderate"
    else:
        return "🟢 Low"


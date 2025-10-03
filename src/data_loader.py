"""
Kannada Fake News Detection - Data Loading and Exploration Module
================================================================

This module provides functions to load, combine, and explore the Kannada fake news dataset.
It handles the specific format of the dataset files and provides comprehensive data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset(data_dir='data'):
    """
    Load and combine fake and real news datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined dataset with text and labels
    """
    data_path = Path(data_dir)
    
    print("📁 Loading Kannada fake news dataset...")
    
    # Load fake news data
    fake_file = data_path / "train_fake.csv"
    real_file = data_path / "train_real.csv"
    
    if not fake_file.exists() or not real_file.exists():
        raise FileNotFoundError("Dataset files not found. Please ensure train_fake.csv and train_real.csv are in the data directory.")
    
    # Read CSV files with proper handling for Kannada text
    try:
        # Read fake file (has header)
        fake_df = pd.read_csv(fake_file, encoding='utf-8')
        # Read real file (no header, need to add column names)
        real_df = pd.read_csv(real_file, names=['headline', 'label'], encoding='utf-8')
        
        # Process fake data - already has proper columns
        fake_processed = pd.DataFrame({
            'text': fake_df['headline'].astype(str),
            'original_label': fake_df['label'],
            'label': 0  # 0 for fake
        })
        
        # Process real data - already has proper columns
        real_processed = pd.DataFrame({
            'text': real_df['headline'].astype(str),
            'original_label': real_df['label'],
            'label': 1  # 1 for real
        })
        
        # Combine datasets
        df = pd.concat([fake_processed, real_processed], ignore_index=True)
        
        # Clean up - remove rows with None text
        df = df.dropna(subset=['text'])
        
        # Remove header rows if they exist
        df = df[df['text'] != 'headline']
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(df)}")
        print(f"   Fake news: {len(fake_processed)} samples")
        print(f"   Real news: {len(real_processed)} samples")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def explore_dataset(df, save_plots=True, output_dir='results'):
    """
    Comprehensive exploration of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to explore
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n🔍 DATASET EXPLORATION")
    print("=" * 50)
    
    # Basic statistics
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📊 Features: {list(df.columns)}")
    
    # Label distribution
    label_counts = df['label'].value_counts()
    print(f"\n📈 Label Distribution:")
    print(f"   Real News (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   Fake News (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\n📏 Text Statistics:")
    print(f"   Average text length: {df['text_length'].mean():.1f} characters")
    print(f"   Average word count: {df['word_count'].mean():.1f} words")
    print(f"   Min text length: {df['text_length'].min()}")
    print(f"   Max text length: {df['text_length'].max()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Label distribution
    label_counts.plot(kind='bar', ax=axes[0,0], color=['#ff6b6b', '#4ecdc4'])
    axes[0,0].set_title('Distribution of Fake vs Real News', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Label (0=Fake, 1=Real)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # 2. Text length distribution
    sns.histplot(data=df, x='text_length', hue='label', bins=50, ax=axes[0,1])
    axes[0,1].set_title('Text Length Distribution by Label', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Text Length (characters)')
    
    # 3. Word count distribution  
    sns.boxplot(data=df, x='label', y='word_count', ax=axes[1,0])
    axes[1,0].set_title('Word Count Distribution by Label', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Label (0=Fake, 1=Real)')
    axes[1,0].set_ylabel('Word Count')
    
    # 4. Text length vs Word count scatter
    scatter = axes[1,1].scatter(df['text_length'], df['word_count'], 
                               c=df['label'], alpha=0.6, cmap='viridis')
    axes[1,1].set_title('Text Length vs Word Count', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Text Length (characters)')
    axes[1,1].set_ylabel('Word Count')
    plt.colorbar(scatter, ax=axes[1,1], label='Label')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_path / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to {output_path / 'dataset_overview.png'}")
    
    plt.show()
    
    # Additional analysis by label
    print(f"\n📊 Detailed Statistics by Label:")
    for label in [0, 1]:
        label_name = "Fake" if label == 0 else "Real"
        subset = df[df['label'] == label]
        print(f"\n{label_name} News:")
        print(f"   Count: {len(subset)}")
        print(f"   Avg text length: {subset['text_length'].mean():.1f}")
        print(f"   Avg word count: {subset['word_count'].mean():.1f}")
        print(f"   Text length std: {subset['text_length'].std():.1f}")
    
    return df


def kannada_text_analysis(df, save_plots=True, output_dir='results'):
    """
    Analyze Kannada-specific text properties.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    save_plots : bool
        Whether to save plots
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n🔤 KANNADA TEXT ANALYSIS")
    print("=" * 50)
    
    # Check for Kannada unicode range (U+0C80 to U+0CFF)
    kannada_pattern = r'[\u0C80-\u0CFF]'
    df['has_kannada'] = df['text'].str.contains(kannada_pattern, regex=True)
    df['kannada_char_count'] = df['text'].str.count(kannada_pattern)
    df['kannada_ratio'] = df['kannada_char_count'] / df['text_length']
    
    print(f"📝 Kannada Content Analysis:")
    print(f"   Texts with Kannada characters: {df['has_kannada'].sum()} ({df['has_kannada'].mean()*100:.1f}%)")
    print(f"   Average Kannada characters per text: {df['kannada_char_count'].mean():.1f}")
    print(f"   Average Kannada ratio: {df['kannada_ratio'].mean():.3f}")
    
    # Check for mixed language content
    english_pattern = r'[a-zA-Z]'
    df['has_english'] = df['text'].str.contains(english_pattern)
    df['is_mixed'] = df['has_kannada'] & df['has_english']
    
    print(f"   Texts with English characters: {df['has_english'].sum()} ({df['has_english'].mean()*100:.1f}%)")
    print(f"   Mixed language texts: {df['is_mixed'].sum()} ({df['is_mixed'].mean()*100:.1f}%)")
    
    # Create language analysis plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Kannada ratio distribution
    sns.histplot(data=df, x='kannada_ratio', hue='label', bins=30, ax=axes[0])
    axes[0].set_title('Kannada Character Ratio Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Kannada Characters / Total Characters')
    
    # Language mixing analysis
    lang_mix = df.groupby(['has_kannada', 'has_english', 'label']).size().unstack(fill_value=0)
    lang_mix.plot(kind='bar', ax=axes[1], color=['#ff6b6b', '#4ecdc4'])
    axes[1].set_title('Language Mixing Patterns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('(Kannada, English)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(['Fake', 'Real'])
    
    # Kannada character count by label
    sns.boxplot(data=df, x='label', y='kannada_char_count', ax=axes[2])
    axes[2].set_title('Kannada Character Count by Label', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Label (0=Fake, 1=Real)')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_path / 'kannada_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Kannada analysis plots saved to {output_path / 'kannada_analysis.png'}")
    
    plt.show()
    
    return df


def generate_wordclouds(df, save_plots=True, output_dir='results'):
    """
    Generate word clouds for fake and real news.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    save_plots : bool
        Whether to save plots
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n☁️ GENERATING WORD CLOUDS")
    print("=" * 50)
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        for i, (label, title) in enumerate([(0, 'Fake News'), (1, 'Real News')]):
            text_data = ' '.join(df[df['label'] == label]['text'].astype(str))
            
            # Create word cloud with Kannada font support
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                collocations=False,
                font_path=None,  # Use default font
                relative_scaling=0.5
            ).generate(text_data)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{title} Word Cloud', fontsize=16, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_path / 'wordclouds.png', dpi=300, bbox_inches='tight')
            print(f"☁️ Word clouds saved to {output_path / 'wordclouds.png'}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not generate word clouds: {e}")


def save_processed_data(df, output_file='data/processed_dataset.csv'):
    """
    Save the processed dataset for later use.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataset
    output_file : str
        Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    # Save only essential columns
    essential_cols = ['text', 'label', 'text_length', 'word_count']
    save_df = df[essential_cols].copy()
    
    save_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 Processed dataset saved to {output_path}")
    print(f"   Shape: {save_df.shape}")
    

if __name__ == "__main__":
    # Load and explore the dataset
    df = load_dataset('data')
    df = explore_dataset(df, save_plots=True)
    df = kannada_text_analysis(df, save_plots=True)
    generate_wordclouds(df, save_plots=True)
    save_processed_data(df)
    
    print("\n🎉 Dataset exploration completed successfully!")
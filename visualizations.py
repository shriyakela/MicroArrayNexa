"""
Visualization functions for microarray spot analysis
Converts R ggplot2 code to Python matplotlib/seaborn equivalents
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f5f5f5'

def create_density_plot(mean_intensities, output_path):
    """
    Create density plot of spot intensities with Z-score normalization
    
    Args:
        mean_intensities: list of mean intensity values
        output_path: path to save the plot
    """
    # Calculate Z-scores
    data = np.array(mean_intensities)
    z_scores = (data - np.mean(data)) / np.std(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot density
    ax.hist(z_scores, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    
    # Add KDE line
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(z_scores)
    x_range = np.linspace(z_scores.min(), z_scores.max(), 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Density')
    
    ax.set_xlabel('Z-score Intensity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Density of Spot Intensities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_boxplot(mean_intensities, output_path, spots_per_region=20):
    """
    Create region-wise boxplots of normalized intensities
    
    Args:
        mean_intensities: list of mean intensity values
        output_path: path to save the plot
        spots_per_region: number of spots per region
    """
    # Create dataframe
    spot_ids = np.arange(1, len(mean_intensities) + 1)
    data = pd.DataFrame({
        'Spot_ID': spot_ids,
        'Mean_Intensity': mean_intensities
    })
    
    # Create regions
    data['Region'] = 'Region_' + (np.ceil(data['Spot_ID'] / spots_per_region)).astype(int).astype(str)
    
    # Calculate Z-scores
    data['Zscore'] = (data['Mean_Intensity'] - data['Mean_Intensity'].mean()) / data['Mean_Intensity'].std()
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    regions = sorted(data['Region'].unique())
    box_data = [data[data['Region'] == region]['Zscore'].values for region in regions]
    
    bp = ax.boxplot(box_data, labels=regions, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Intensity (Z-score)', fontsize=12, fontweight='bold')
    ax.set_title('Protein Microarray Region-wise Boxplots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_heatmap(mean_intensities, output_path, spots_per_region=20):
    """
    Create heatmap of spot intensities across regions
    
    Args:
        mean_intensities: list of mean intensity values
        output_path: path to save the plot
        spots_per_region: number of spots per region
    """
    # Create dataframe
    spot_ids = np.arange(1, len(mean_intensities) + 1)
    data = pd.DataFrame({
        'Spot_ID': spot_ids,
        'Mean_Intensity': mean_intensities
    })
    
    # Create regions
    data['Region'] = 'Region_' + (np.ceil(data['Spot_ID'] / spots_per_region)).astype(int).astype(str)
    
    # Calculate Z-scores
    data['Zscore'] = (data['Mean_Intensity'] - data['Mean_Intensity'].mean()) / data['Mean_Intensity'].std()
    
    # Create pivot table for heatmap
    regions = sorted(data['Region'].unique())
    heatmap_data = []
    for region in regions:
        region_data = data[data['Region'] == region]['Zscore'].values
        heatmap_data.append(region_data)
    
    # Pad to equal length
    max_len = max(len(row) for row in heatmap_data)
    heatmap_array = np.full((len(regions), max_len), np.nan)
    for i, row in enumerate(heatmap_data):
        heatmap_array[i, :len(row)] = row
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(heatmap_array, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('Spot ID (within Region)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Region', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap of Microarray Spot Intensities', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score Intensity', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_intensity_plot(mean_intensities, output_path, spots_per_region=20):
    """
    Create region-wise mean intensity bar plot with error bars
    
    Args:
        mean_intensities: list of mean intensity values
        output_path: path to save the plot
        spots_per_region: number of spots per region
    """
    # Create dataframe
    spot_ids = np.arange(1, len(mean_intensities) + 1)
    data = pd.DataFrame({
        'Spot_ID': spot_ids,
        'Mean_Intensity': mean_intensities
    })
    
    # Create regions
    data['Region'] = 'Region_' + (np.ceil(data['Spot_ID'] / spots_per_region)).astype(int).astype(str)
    
    # Calculate Z-scores
    data['Zscore'] = (data['Mean_Intensity'] - data['Mean_Intensity'].mean()) / data['Mean_Intensity'].std()
    
    # Group by region and calculate statistics
    region_summary = data.groupby('Region')['Zscore'].agg(['mean', 'std']).reset_index()
    region_summary.columns = ['Region', 'Mean_Intensity', 'SD']
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(region_summary))
    bars = ax.bar(x_pos, region_summary['Mean_Intensity'], 
                  color=plt.cm.Set2(np.linspace(0, 1, len(region_summary))),
                  edgecolor='black', linewidth=1.2)
    
    ax.errorbar(x_pos, region_summary['Mean_Intensity'], 
                yerr=region_summary['SD'], 
                fmt='none', color='black', capsize=5, capthick=2)
    
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Z-score', fontsize=12, fontweight='bold')
    ax.set_title('Region-wise Mean Intensity', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(region_summary['Region'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(mean_intensities, result_folder):
    """
    Generate all four visualizations
    
    Args:
        mean_intensities: list of mean intensity values
        result_folder: folder to save visualizations
        
    Returns:
        dict with paths to all visualization files
    """
    os.makedirs(result_folder, exist_ok=True)
    
    density_path = os.path.join(result_folder, 'density_plot.png')
    boxplot_path = os.path.join(result_folder, 'boxplot.png')
    heatmap_path = os.path.join(result_folder, 'heatmap.png')
    intensity_path = os.path.join(result_folder, 'intensity_plot.png')
    
    create_density_plot(mean_intensities, density_path)
    create_boxplot(mean_intensities, boxplot_path)
    create_heatmap(mean_intensities, heatmap_path)
    create_intensity_plot(mean_intensities, intensity_path)
    
    return {
        'density': density_path.replace("\\", "/"),
        'boxplot': boxplot_path.replace("\\", "/"),
        'heatmap': heatmap_path.replace("\\", "/"),
        'intensity': intensity_path.replace("\\", "/")
    }

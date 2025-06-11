import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("Set1")

print("Reading and processing Excel data...")

# Read the Excel file
df = pd.read_excel('stats.xlsx')
print("Original data structure:")
print(df.to_string())

# Extract the data manually based on the visible structure
# Row 1 contains the metric headers
# Rows 2+ contain the model data

# Create a proper dataframe structure
data = {
    'Model': [],
    'ATE_MacroF1': [],
    'ASC_Positive': [],
    'ASC_Neutral': [],
    'ASC_Negative': [],
    'ASC_Macro': []
}

# Extract data from rows 2-6 (indices 2-6)
model_rows = df.iloc[2:7]  # greekBert through xlmRoBERTa

for idx, row in model_rows.iterrows():
    model_name = row.iloc[0]  # First column is model name
    ate_f1 = row.iloc[1]      # ATE macro F1
    asc_pos = row.iloc[3]     # ASC Positive F1
    asc_neu = row.iloc[4]     # ASC Neutral F1  
    asc_neg = row.iloc[5]     # ASC Negative F1
    asc_macro = row.iloc[6]   # ASC Macro F1
    
    # Clean model name
    model_name = str(model_name).strip()
    
    data['Model'].append(model_name)
    data['ATE_MacroF1'].append(float(ate_f1) if pd.notna(ate_f1) else 0)
    data['ASC_Positive'].append(float(asc_pos) if pd.notna(asc_pos) else 0)
    data['ASC_Neutral'].append(float(asc_neu) if pd.notna(asc_neu) else 0)
    data['ASC_Negative'].append(float(asc_neg) if pd.notna(asc_neg) else 0)
    data['ASC_Macro'].append(float(asc_macro) if pd.notna(asc_macro) else 0)

# Create cleaned dataframe
clean_df = pd.DataFrame(data)
print("\nCleaned data:")
print(clean_df.to_string())

# Define colors for each model
model_colors = {
    'greekBert': '#1f77b4',
    'greekBert lemma': '#ff7f0e', 
    'greekBert lemma augmented': '#2ca02c',
    'ATE and ASC greekBert': '#d62728',
    'ATE and ASC xlmRoBERTa': '#9467bd'
}

# Create the main comparison plot
fig, ax = plt.subplots(figsize=(14, 8))

# Metrics to plot
metrics = ['ATE_MacroF1', 'ASC_Positive', 'ASC_Neutral', 'ASC_Negative', 'ASC_Macro']
metric_labels = ['ATE Macro F1', 'ASC Positive F1', 'ASC Neutral F1', 'ASC Negative F1', 'ASC Macro F1']

# Set up the bar positions
x = np.arange(len(metrics))
width = 0.15  # Width of bars
n_models = len(clean_df)

# Colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create bars for each model
for i, (idx, row) in enumerate(clean_df.iterrows()):
    model_name = row['Model']
    values = [row[metric] for metric in metrics]
    
    # Create bars
    bars = ax.bar(x + i * width, values, width, 
                  label=model_name, color=colors[i % len(colors)], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        if value > 0:  # Only label non-zero values
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold', rotation=0)

# Customize the plot
ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - F1 Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels(metric_labels, rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)  # F1 scores are between 0 and 1

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('model_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('model_f1_comparison.pdf', bbox_inches='tight')

print("\nMain comparison plot saved as 'model_f1_comparison.png' and 'model_f1_comparison.pdf'")

# Create a focused plot for just the Macro F1 scores
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Macro F1 metrics
macro_metrics = ['ATE_MacroF1', 'ASC_Macro']
macro_labels = ['ATE Macro F1', 'ASC Macro F1']

x_macro = np.arange(len(macro_metrics))

for i, (idx, row) in enumerate(clean_df.iterrows()):
    model_name = row['Model']
    macro_values = [row[metric] for metric in macro_metrics]
    
    bars = ax2.bar(x_macro + i * width, macro_values, width, 
                  label=model_name, color=colors[i % len(colors)], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, value in zip(bars, macro_values):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')

ax2.set_xlabel('Macro F1 Metrics', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax2.set_title('Macro F1 Score Comparison - ATE vs ASC', fontsize=14, fontweight='bold')
ax2.set_xticks(x_macro + width * (n_models - 1) / 2)
ax2.set_xticklabels(macro_labels)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('macro_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('macro_f1_comparison.pdf', bbox_inches='tight')

print("Macro F1 comparison plot saved as 'macro_f1_comparison.png' and 'macro_f1_comparison.pdf'")

# Create a sentiment-specific F1 comparison (ASC only)
fig3, ax3 = plt.subplots(figsize=(10, 6))

sentiment_metrics = ['ASC_Positive', 'ASC_Neutral', 'ASC_Negative']
sentiment_labels = ['Positive F1', 'Neutral F1', 'Negative F1']

x_sent = np.arange(len(sentiment_metrics))

for i, (idx, row) in enumerate(clean_df.iterrows()):
    model_name = row['Model']
    sent_values = [row[metric] for metric in sentiment_metrics]
    
    bars = ax3.bar(x_sent + i * width, sent_values, width, 
                  label=model_name, color=colors[i % len(colors)], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, value in zip(bars, sent_values):
        if value > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')

ax3.set_xlabel('Sentiment Classes', fontsize=12, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax3.set_title('Sentiment-Specific F1 Score Comparison (ASC)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_sent + width * (n_models - 1) / 2)
ax3.set_xticklabels(sentiment_labels)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('sentiment_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('sentiment_f1_comparison.pdf', bbox_inches='tight')

print("Sentiment F1 comparison plot saved as 'sentiment_f1_comparison.png' and 'sentiment_f1_comparison.pdf'")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Number of models compared: {len(clean_df)}")
print(f"Models: {clean_df['Model'].tolist()}")
print(f"\nBest ATE Macro F1: {clean_df['ATE_MacroF1'].max():.3f}")
print(f"Best ASC Macro F1: {clean_df['ASC_Macro'].max():.3f}")
print(f"Best model for ATE: {clean_df.loc[clean_df['ATE_MacroF1'].idxmax(), 'Model']}")
print(f"Best model for ASC: {clean_df.loc[clean_df['ASC_Macro'].idxmax(), 'Model']}")

plt.show()
print("\nVisualization complete!") 
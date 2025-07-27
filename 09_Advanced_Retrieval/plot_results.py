#plot_results.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## Final Notes
'''
All visualizations above provide different perspectives on the evaluation results:
- **Bar Chart**: Direct comparison of individual metrics with error bars
- **Heatmap**: Quick visual comparison across all metrics and strategies  
- **Radar Chart**: Overall performance profile for each strategy
- **Performance Charts**: Cost and latency analysis
- **Effectiveness vs Efficiency**: Strategic positioning analysis

Use these visualizations to make informed decisions about which retrieval strategy best fits your requirements.
'''

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create a comprehensive comparison plot
def plot_evaluation_metrics(eval_summary_results):
    """
    Plot evaluation metrics for all chains with error bars
    """
    # Extract data for plotting
    chains = list(eval_summary_results.keys())
    metrics = ['LLMContextPrecisionWithReference', 'LLMContextRecall', 'ContextEntityRecall']
    
    # Prepare data
    plot_data = []
    for chain in chains:
        means = eval_summary_results[chain]["means"]
        stds = eval_summary_results[chain]["stds"]
        
        for metric in metrics:
            plot_data.append({
                'Chain': chain,
                'Metric': metric.replace('LLM', '').replace('WithReference', ''),  # Shorter names
                'Mean': means[metric],
                'Std': stds[metric]
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grouped bar chart
    x_pos = range(len(chains))
    width = 0.25
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, metric in enumerate(df['Metric'].unique()):
        metric_data = df[df['Metric'] == metric]
        means = metric_data['Mean'].values
        stds = metric_data['Std'].values
        
        positions = [x + width * i for x in x_pos]
        bars = ax.bar(positions, means, width, yerr=stds, 
                     label=metric, color=colors[i], alpha=0.8,
                     capsize=5, error_kw={'linewidth': 2})
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[j] + 0.01,
                   f'{means[j]:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Retrieval Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('RAG Evaluation Metrics Comparison\n(Higher is Better)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([x + width for x in x_pos])
    ax.set_xticklabels([chain.replace('_', ' ').title() for chain in chains])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Create metrics comparison table
def create_metrics_table(eval_summary_results):
    """
    Create a detailed table of all metrics
    """
    table_data = []
    for chain in eval_summary_results.keys():
        means = eval_summary_results[chain]["means"]
        stds = eval_summary_results[chain]["stds"]
        
        table_data.append({
            'Chain': chain.replace('_', ' ').title(),
            'Context Precision': f"{means['LLMContextPrecisionWithReference']:.3f} ± {stds['LLMContextPrecisionWithReference']:.3f}",
            'Context Recall': f"{means['LLMContextRecall']:.3f} ± {stds['LLMContextRecall']:.3f}",
            'Entity Recall': f"{means['ContextEntityRecall']:.3f} ± {stds['ContextEntityRecall']:.3f}"
        })
    
    df_table = pd.DataFrame(table_data)
    
    print("\\n" + "="*80)
    print("DETAILED METRICS TABLE")
    print("="*80)
    print(df_table.to_string(index=False))
    
    return df_table


# Create heatmap visualization
def create_heatmap_visualization(eval_summary_results):
    """
    Create a heatmap showing metric performance across chains
    """
    # Prepare data for heatmap
    heatmap_data = []
    chains = list(eval_summary_results.keys())
    metrics = ['LLMContextPrecisionWithReference', 'LLMContextRecall', 'ContextEntityRecall']
    
    for chain in chains:
        means = eval_summary_results[chain]["means"]
        row_data = [means[metric] for metric in metrics]
        heatmap_data.append(row_data)
    
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(
        heatmap_data, 
        index=[chain.replace('_', ' ').title() for chain in chains],
        columns=['Context Precision', 'Context Recall', 'Entity Recall']
    )
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap, 
                annot=True, 
                cmap='RdYlBu_r',
                fmt='.3f',
                linewidths=0.5,
                cbar_kws={'label': 'Score'},
                square=True)
    
    plt.title('RAG Evaluation Metrics Heatmap\\n(Darker = Better Performance)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Retrieval Strategies', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return df_heatmap

# Create performance comparison radar chart
def create_radar_chart(eval_summary_results):
    """
    Create a radar chart comparing all metrics across chains
    """
    import numpy as np
    
    chains = list(eval_summary_results.keys())
    metrics = ['LLMContextPrecisionWithReference', 'LLMContextRecall', 'ContextEntityRecall']
    metric_labels = ['Context\\nPrecision', 'Context\\nRecall', 'Entity\\nRecall']
    
    # Number of metrics
    N = len(metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#E71D36']
    
    for i, chain in enumerate(chains):
        means = eval_summary_results[chain]["means"]
        values = [means[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=chain.replace('_', ' ').title(), color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Customize the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('RAG Evaluation Metrics - Radar Chart\\n(Further from center = Better)', 
              fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.show()

# Generate alternative visualizations
#print("\\nCreating heatmap visualization...")
#heatmap_df = create_heatmap_visualization(eval_summary_results)

#print("\\nCreating radar chart...")
#create_radar_chart(eval_summary_results)


# Performance metrics visualization
def plot_performance_metrics(eval_langsmith_summary_results):
    """
    Plot cost and latency performance metrics
    """
    chains = list(eval_langsmith_summary_results.keys())
    latencies = [eval_langsmith_summary_results[chain]["average_latency"] for chain in chains]
    costs = [eval_langsmith_summary_results[chain]["total_cost"] for chain in chains]
    
    # Create subplot for cost and latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Latency plot
    bars1 = ax1.bar(chains, latencies, color='#2E86AB', alpha=0.8)
    ax1.set_title('Average Latency by Retrieval Strategy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Retrieval Strategy', fontsize=12)
    ax1.set_ylabel('Latency (seconds)', fontsize=12)
    ax1.set_xticklabels([chain.replace('_', ' ').title() for chain in chains], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, latency in zip(bars1, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{latency:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # Cost plot
    bars2 = ax2.bar(chains, costs, color='#A23B72', alpha=0.8)
    ax2.set_title('Total Cost by Retrieval Strategy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Retrieval Strategy', fontsize=12)
    ax2.set_ylabel('Total Cost ($)', fontsize=12)
    ax2.set_xticklabels([chain.replace('_', ' ').title() for chain in chains], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cost in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'${cost:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Create performance summary table
    perf_table_data = []
    for chain in chains:
        results = eval_langsmith_summary_results[chain]
        perf_table_data.append({
            'Chain': chain.replace('_', ' ').title(),
            'Avg Latency (s)': f"{results['average_latency']:.3f}",
            'Total Cost ($)': f"{results['total_cost']:.4f}",
            'Cost per Query ($)': f"{results['total_cost']/50:.6f}"  # Assuming 50 queries
        })
    
    perf_df = pd.DataFrame(perf_table_data)
    print("\\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    print(perf_df.to_string(index=False))
    
    return perf_df

# Create combined effectiveness vs efficiency plot
def plot_effectiveness_vs_efficiency(eval_summary_results, eval_langsmith_summary_results):
    """
    Create a scatter plot showing effectiveness vs efficiency
    """
    chains = list(eval_summary_results.keys())
    
    # Calculate overall effectiveness score (average of all RAGAS metrics)
    effectiveness_scores = []
    for chain in chains:
        means = eval_summary_results[chain]["means"]
        avg_score = (means['LLMContextPrecisionWithReference'] + 
                    means['LLMContextRecall'] + 
                    means['ContextEntityRecall']) / 3
        effectiveness_scores.append(avg_score)
    
    # Get efficiency metrics (inverse of latency for "efficiency")
    latencies = [eval_langsmith_summary_results[chain]["average_latency"] for chain in chains]
    costs = [eval_langsmith_summary_results[chain]["total_cost"] for chain in chains]
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with cost as bubble size
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#E71D36']
    
    for i, chain in enumerate(chains):
        plt.scatter(effectiveness_scores[i], 1/latencies[i], 
                   s=costs[i]*50000, alpha=0.6, 
                   color=colors[i % len(colors)],
                   label=f"{chain.replace('_', ' ').title()}\\n(Cost: ${costs[i]:.4f})")
    
    plt.xlabel('Effectiveness Score\\n(Average of RAGAS Metrics)', fontsize=12, fontweight='bold')
    plt.ylabel('Efficiency Score\\n(1/Average Latency)', fontsize=12, fontweight='bold')
    plt.title('Effectiveness vs Efficiency Analysis\\n(Bubble size = Total Cost)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for each point
    for i, chain in enumerate(chains):
        plt.annotate(f'{effectiveness_scores[i]:.3f}', 
                    (effectiveness_scores[i], 1/latencies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()

'''
# Generate performance visualizations
if 'eval_langsmith_summary_results' in globals() and eval_langsmith_summary_results:
    print("Creating performance metrics visualization...")
    perf_df = plot_performance_metrics(eval_langsmith_summary_results)
    
    print("\\nCreating effectiveness vs efficiency analysis...")
    plot_effectiveness_vs_efficiency(eval_summary_results, eval_langsmith_summary_results)
else:
    print("LangSmith results not available yet. Run the LangSmith evaluation section first.")


# Generate the evaluation metrics plot
print("Creating evaluation metrics visualization...")
plot_df = plot_evaluation_metrics(eval_summary_results)

# Display detailed metrics table
metrics_table = create_metrics_table(eval_summary_results)
'''
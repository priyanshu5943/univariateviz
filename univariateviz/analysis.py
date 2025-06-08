 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import entropy
import numpy as np
from IPython.display import display, HTML

feature_print_count = {}

def univariate_analysis_category(col_name, cat_columns):
    global feature_print_count
    series = cat_columns[col_name]
    value_counts = series.value_counts(dropna=False)
    percentage = (value_counts / value_counts.sum()) * 100
    num_unique = value_counts.shape[0]

    total_rows = len(series)
    missing_count = series.isna().sum()
    missing_pct = (missing_count / total_rows) * 100
    most_cat = value_counts.idxmax()
    least_cat = value_counts.idxmin()
    imbalance_ratio = value_counts.max() / max(value_counts.min(), 1)
    ent_val = entropy(value_counts)
    p = value_counts / value_counts.sum()
    gini_index = 1 - (p**2).sum()
    mode_ratio = value_counts.max() / total_rows
    top_n = 1 if num_unique <= 5 else (3 if num_unique <= 20 else 5)

    feature_print_count[col_name] = feature_print_count.get(col_name, 0) + 1
    font_size = 24 + 4 * feature_print_count[col_name]

    display(HTML(f"<div style='margin-bottom:50px;'>"))
    display(HTML(f"<h2 style='text-align:center; font-size:{font_size}px; color:green;'><b>{col_name}</b></h2>"))

    print(f"{'Total Rows':<30}: {total_rows}")
    print(f"{'Unique Categories':<30}: {num_unique}")
    print(f"{'Missing Values':<30}: {missing_count} ({missing_pct:.2f}%)")
    print(f"{'Most Frequent Category':<30}: {most_cat} ({value_counts.max()} | {percentage[most_cat]:.2f}%)")
    print(f"{'Least Frequent Category':<30}: {least_cat} ({value_counts.min()} | {percentage[least_cat]:.2f}%)")
    print(f"{'Imbalance Ratio (max/min)':<30}: {imbalance_ratio:.2f}")
    print(f"{'Entropy':<30}: {ent_val:.2f}")
    print(f"{'Gini Index':<30}: {gini_index:.2f}")
    print(f"{'Mode Frequency Ratio':<30}: {mode_ratio:.2f}")

    top_html = "<ul>" + "".join(
        [f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>" for i, (k, v) in enumerate(value_counts.head(top_n).items())]) + "</ul>"

    bottom_html = "<ul>" + "".join(
        [f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>" for i, (k, v) in enumerate(value_counts.tail(top_n).items())]) + "</ul>"

    table_html = f"""
    <table style='width:100%; font-size:14px; table-layout:fixed; margin-top:10px;'>
      <tr>
        <th style='text-align:center; width:50%; color:#00CED1;'>Top Categories</th>
        <th style='text-align:center; width:50%; color:#FF69B4;'>Bottom Categories</th>
      </tr>
      <tr>
        <td style='vertical-align:top; padding: 10px;'>{top_html}</td>
        <td style='vertical-align:top; padding: 10px;'>{bottom_html}</td>
      </tr>
    </table>
    """
    display(HTML(table_html))

    colors = ['#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',  
              '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',  
              '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB', 
              '#32CD32', '#00CED1', '#1E90FF', "#FBFB14", '#7CFC00']

    rotate_labels = -45 if len(value_counts) > 8 else 0

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"{col_name} - Count", f"{col_name} - % Share"),
        specs=[[{"type": "bar"}, {"type": "domain"}]]
    )

    fig.add_trace(
        go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            text=value_counts.values,
            textposition='outside',
            marker_color=colors[:len(value_counts)],
            name=f'{col_name} Count',
            hovertemplate=f'{col_name}: %{{x}}<br>Count: %{{y}}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=value_counts.index.astype(str),
            values=percentage,
            hole=0.45,
            marker_colors=colors[:len(value_counts)],
            name=f'{col_name} % Share',
            hovertemplate=f'{col_name}: %{{label}}<br>Share: %{{percent}}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text=f"Distribution of {col_name}",
        title_font=dict(size=20, family='Arial'),
        showlegend=False,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=11),
        height=300,
        width=670,
        margin=dict(t=50, b=30, l=30, r=30),
        xaxis_tickangle=rotate_labels
    )

    fig.show()
    display(HTML("</div>"))

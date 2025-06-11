 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import entropy
import numpy as np
from IPython.display import display, HTML



feature_print_count = {}

def analyze_categorical_univariate(df):
    def is_categorical(col):
        return (
            df[col].dtype == 'object' or 
            df[col].dtype.name == 'category' or 
            (np.issubdtype(df[col].dtype, np.number) and df[col].nunique() <= 15)
        )

    def summarize_categorical(series, feature):
        global feature_print_count

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

        feature_print_count[feature] = feature_print_count.get(feature, 0) + 1
        font_size = 24 + 4 * feature_print_count[feature]

        display(HTML(f"<div style='margin-bottom:50px;'>"))
        display(HTML(f"<h2 style='text-align:center; font-size:{font_size}px; color:green;'><b>{feature}</b></h2>"))

        print(f"{'Total Rows':<30}: {total_rows}")
        print(f"{'Unique Categories':<30}: {num_unique}")
        print(f"{'Missing Values':<30}: {missing_count} ({missing_pct:.2f}%)")
        print(f"{'Most Frequent Category':<30}: {most_cat} ({value_counts.max()} | {percentage[most_cat]:.2f}%)")
        print(f"{'Least Frequent Category':<30}: {least_cat} ({value_counts.min()} | {percentage[least_cat]:.2f}%)")
        print(f"{'Imbalance Ratio (max/min)':<30}: {imbalance_ratio:.2f}")
        print(f"{'Entropy':<30}: {ent_val:.2f}")
        print(f"{'Gini Index':<30}: {gini_index:.2f}")
        print(f"{'Mode Frequency Ratio':<30}: {mode_ratio:.2f}")

        top_html = "<ul>" + "".join([f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>" for i, (k, v) in enumerate(value_counts.head(top_n).items())]) + "</ul>"
        bottom_html = "<ul>" + "".join([f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>" for i, (k, v) in enumerate(value_counts.tail(top_n).items())]) + "</ul>"

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

        return value_counts, percentage

    def plot_categorical_distribution(value_counts, percentage, feature):
        colors = [
            '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',
            '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',
            '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB',
            '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
        ]

        rotate_labels = -45 if len(value_counts) > 8 else 0

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{feature} - Count", f"{feature} - % Share"),
            specs=[[{"type": "bar"}, {"type": "domain"}]]
        )

        fig.add_trace(
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                text=value_counts.values,
                textposition='outside',
                marker_color=colors[:len(value_counts)],
                name=f'{feature} Count',
                hovertemplate=f'{feature}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Pie(
                labels=value_counts.index.astype(str),
                values=percentage,
                hole=0.45,
                marker_colors=colors[:len(value_counts)],
                name=f'{feature} % Share',
                hovertemplate=f'{feature}: %{{label}}<br>Share: %{{percent}}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title_text=f"Distribution of {feature}",
            title_font=dict(size=20, family='Arial'),
            showlegend=False,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='white', size=11),
            height=320,
            width=670,
            margin=dict(t=50, b=30, l=30, r=30),
            xaxis_tickangle=rotate_labels
        )

        fig.show()
        display(HTML("</div>"))

    for feature in df.columns:
        if feature.lower() == 'id':
            continue
        if is_categorical(feature):
            series = df[feature]
            value_counts, percentage = summarize_categorical(series, feature)
            plot_categorical_distribution(value_counts, percentage, feature)



import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew, kurtosis
from IPython.display import display, HTML

def analyze_numeric_univariate(df):
    id_like_keywords = {'id', 'index', 'serial'}
    exclude_cols = {col for col in df.columns if col.lower() in id_like_keywords}
    numeric_cols = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

    feature_print_count = {}

    def create_combined_plot(data, feature, color, width=670, height=310):
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=('Histogram', 'Violin Plot')
        )

        fig.add_trace(
            go.Histogram(
                x=data[feature],
                marker=dict(color=color),
                name='Histogram'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Violin(
                y=data[feature],
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                name='Violin',
                orientation='v'
            ),
            row=1, col=2
        )

        annotations = [
            dict(text=f"<b>histogram</b>", x=0.22, y=1.1, xref='paper', yref='paper', showarrow=False, font=dict(size=16, color=color)),
            dict(text=f"<b>violin_plot</b>", x=0.78, y=1.1, xref='paper', yref='paper', showarrow=False, font=dict(size=16, color=color))
        ]

        fig.update_layout(
            title_text=f"Distribution of {feature}",
            title_font=dict(size=20, family='Arial'),
            showlegend=False,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            width=width,
            height=height,
            annotations=annotations
        )

        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False)
        fig.show()

    def numeric_feature_summary(df, feature):
        nonlocal feature_print_count

        series = df[feature]
        clean_series = series.dropna()

        total_rows = len(series)
        missing_count = series.isna().sum()
        missing_pct = (missing_count / total_rows) * 100

        min_val = clean_series.min()
        max_val = clean_series.max()
        mean_val = clean_series.mean()
        median_val = clean_series.median()
        std_val = clean_series.std()
        skew_val = skew(clean_series)
        kurt_val = kurtosis(clean_series)
        mode_val = clean_series.mode().iloc[0] if not clean_series.mode().empty else "N/A"

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)].count()

        cv = std_val / mean_val if mean_val != 0 else float("nan")

        feature_print_count[feature] = feature_print_count.get(feature, 0) + 1
        font_size = 24 + 4 * feature_print_count[feature]

        display(HTML(f"<div style='margin-bottom:50px;'>"))
        display(HTML(f"<h2 style='text-align:center; font-size:{font_size}px; color:green;'><b>{feature}</b></h2>"))

        summary_data = [
            ("Total Rows", f"{total_rows}"),
            ("Missing Values", f"{missing_count} ({missing_pct:.2f}%)"),
            ("Mean", f"{mean_val:.2f}"),
            ("Median", f"{median_val:.2f}"),
            ("Standard Deviation", f"{std_val:.2f}"),
            ("Min", f"{min_val:.2f}"),
            ("Max", f"{max_val:.2f}"),
            ("Skewness", f"{skew_val:.2f}"),
            ("Kurtosis", f"{kurt_val:.2f}"),
            ("Mode", f"{mode_val}"),
            ("Q1 (25%)", f"{q1:.2f}"),
            ("Q3 (75%)", f"{q3:.2f}"),
            ("IQR", f"{iqr:.2f}"),
            ("Lower Bound (1.5*IQR)", f"{lower_bound:.2f}"),
            ("Upper Bound (1.5*IQR)", f"{upper_bound:.2f}"),
            ("Outlier Count (1.5*IQR)", f"{outlier_count}"),
            ("Coefficient of Variation", f"{cv:.2f}"),
        ]

        half = len(summary_data) // 2
        col1 = summary_data[:half]
        col2 = summary_data[half:]

        col1_html = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in col1])
        col2_html = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in col2])

        table_html = f"""
        <table style='width:100%; font-size:14px; table-layout:fixed; margin-top:10px;'>
          <tr>
            <th style='text-align:center; width:50%; color:#00CED1;'>Summary</th>
            <th style='text-align:center; width:50%; color:#FF69B4;'>Details</th>
          </tr>
          <tr>
            <td style='vertical-align:top; padding: 10px;'><ul>{col1_html}</ul></td>
            <td style='vertical-align:top; padding: 10px;'><ul>{col2_html}</ul></td>
          </tr>
        </table>
        """
        display(HTML(table_html))

    color_palette = [
        '#FFD700', '#FFA500', '#00FA9A', '#FFB6C1', '#FF1493',
        'red', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
    ]

    for i, feature in enumerate(numeric_cols):
        numeric_feature_summary(df, feature)
        create_combined_plot(df, feature, color_palette[i % len(color_palette)])
        print("\n" * 3)

import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display, HTML
from scipy.stats import f_oneway, kruskal
import itertools
import colorsys

# === Utility Color Functions ===

base_colors = [
    '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#4682B4', 'red',
    '#7CFC00', '#98FB98', '#9370DB', '#32CD32', '#00CED1',
    '#1E90FF', '#FFFF00', '#7CFC00'
]

_single_color_cycle = itertools.cycle(base_colors)

def adjust_color(hex_color, factor=1.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    l = min(1.0, l * factor)
    r_adj, g_adj, b_adj = colorsys.hls_to_rgb(h, l, s)
    return '#{0:02X}{1:02X}{2:02X}'.format(int(r_adj*255), int(g_adj*255), int(b_adj*255))

def get_recycled_colors(categories):
    color_map = {}
    num_base = len(base_colors)
    for i, category in enumerate(categories):
        base_idx = i % num_base
        repeat_idx = i // num_base
        base_color = base_colors[base_idx]
        if repeat_idx == 0:
            color_map[category] = base_color
        else:
            color_map[category] = adjust_color(base_color, 1 + 0.1 * repeat_idx)
    return color_map

def _spacer_div(height=25):
    display(HTML(f"<div style='margin-top:{height}px; margin-bottom:{height}px; border-bottom:1px solid #333;'></div>"))


# === Plotting Functions ===

def _plot_categorical_vs_categorical(df, x_col, color_col):
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {x_col} by {color_col}</b></h2>"))

    unique_categories = df[color_col].dropna().unique()
    category_to_color = get_recycled_colors(unique_categories)
    
    fig = px.histogram(
        df, x=x_col, color=color_col, barmode='group',
        color_discrete_map=category_to_color
    )

    fig.update_layout(
        xaxis_title=x_col, yaxis_title='Count',
        plot_bgcolor='#000000', paper_bgcolor='#000000',
        font=dict(color='white', size=15),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white', showline=False),
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='white', showline=False),
        legend_title_text=color_col,
        legend_font=dict(color='white', size=12),
        width=650, height=300
    )
    fig.show()

def _plot_categorical_vs_numeric(df, cat_col, target_col):
    display(HTML(f"<h2 style='text-align:center; font-size:22px; color:green;'><b>Distribution of {target_col} by {cat_col}</b></h2>"))

    summary_df = df.groupby(cat_col)[target_col].agg(['count', 'mean', 'median', 'std']).reset_index()
    summary_df.columns = [
        cat_col, 'Count', f'Mean of {target_col}', f'Median of {target_col}', f'Std Dev of {target_col}'
    ]
    summary_df = summary_df.sort_values(by=f'Mean of {target_col}', ascending=False)

    groups = [group[target_col].dropna().values for _, group in df.groupby(cat_col)]
    if len(groups) > 1:
        test_stat, p_val = f_oneway(*groups)
        test_name = "ANOVA F-test"
    else:
        test_stat, p_val = (np.nan, np.nan)
        test_name = "Insufficient Groups"

    if np.isnan(p_val) or np.isnan(test_stat):
        test_stat, p_val = kruskal(*groups)
        test_name = "Kruskal-Wallis H-test"

    summary_data = [
        ("Total Categories", f"{summary_df.shape[0]}"),
        ("Overall Target Mean", f"{df[target_col].mean():.2f}"),
        (f"{test_name} Stat", f"{test_stat:.2f}"),
        (f"{test_name} P-value", f"{p_val:.4f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")
    ]

    half = len(summary_data) // 2
    col1 = summary_data[:half]
    col2 = summary_data[half:]
    col1_html = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in col1])
    col2_html = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in col2])

    table_html = f"""
    <table style='width:100%; font-size:14px; table-layout:fixed; margin-top:10px;'>
      <tr>
        <th style='text-align:center; width:50%; color:#00CED1;'>Summary</th>
        <th style='text-align:center; width:50%; color:#FF69B4;'>Details</th>
      </tr>
      <tr>
        <td style='vertical-align:top; padding: 10px;'><ul>{col1_html}</ul></td>
        <td style='vertical-align:top; padding: 10px;'><ul>{col2_html}</ul></td>
      </tr>
    </table>
    """
    display(HTML(table_html))
    display(summary_df.style.background_gradient(cmap='viridis').set_table_attributes('style="width:75%; margin:auto;"'))

    unique_categories = df[cat_col].dropna().unique()
    color_map = {cat: base_colors[j % len(base_colors)] for j, cat in enumerate(unique_categories)}
    fig = px.box(df, x=cat_col, y=target_col, color=cat_col, color_discrete_map=color_map)
    fig.update_layout(
        xaxis_title=cat_col, yaxis_title=target_col,
        plot_bgcolor='black', paper_bgcolor='black',
        font=dict(color='white', size=15),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white', showline=False),
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='white', showline=False),
        legend_title_text=cat_col,
        legend_font=dict(color='white', size=12),
        width=700, height=290
    )
    fig.show()

def _plot_numeric_vs_numeric(df, x_col, y_col, category_col=None):
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    strength = ("strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.3 else "weak")
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "no"

    title = f"{x_col} vs {y_col}"
    if category_col:
        title += f" by {category_col}"

    display(HTML(f"<h2 style='text-align:center; font-size:22px; font-weight:bold; color:green;'>{title}</h2>"))
    display(HTML(f"<p style='font-size:15px; color:#FF69B4;'><b>Overall Correlation:</b> {corr:.2f} ({strength} {direction})</p>"))

    if category_col:
        cats = df[category_col].astype(str).unique()
        color_seq = [next(_single_color_cycle) for _ in cats]
        fig = px.scatter(df, x=x_col, y=y_col, color=category_col, color_discrete_sequence=color_seq)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=[next(_single_color_cycle)])

    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col,
        plot_bgcolor='#000000', paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        xaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5, zeroline=False, color='white', showline=True, linewidth=1, linecolor='white', ticks='outside', tickcolor='white'),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5, zeroline=False, color='white', showline=True, linewidth=1, linecolor='white', ticks='outside', tickcolor='white'),
        legend_title_text=(category_col if category_col else None),
        legend_font=dict(color='white', size=12),
        width=650, height=320
    )
    fig.show()

# === Dispatcher Function ===
def analyze_feature_target_relationships(df, target_col, color_col=None):
    heading_color = "#FF6347"
    cat_vs_num = []
    cat_vs_cat = []
    num_vs_num = []

    # Identify and exclude id column if present (case-insensitive)
    id_cols = [col for col in df.columns if col.lower() == 'id']
    excluded_cols = set(id_cols + [target_col])

    num_target_is_cat = pd.api.types.is_categorical_dtype(df[target_col]) or df[target_col].dtype == object

    for feature in df.columns:
        if feature in excluded_cols:
            continue

        feature_is_cat = pd.api.types.is_categorical_dtype(df[feature]) or df[feature].dtype == object
        feature_is_num = pd.api.types.is_numeric_dtype(df[feature])

        if feature_is_num and pd.api.types.is_numeric_dtype(df[target_col]):
            num_vs_num.append(feature)
        elif feature_is_cat and not num_target_is_cat:
            cat_vs_num.append(feature)
        elif feature_is_cat and num_target_is_cat:
            cat_vs_cat.append(feature)
        elif feature_is_num and num_target_is_cat:
            cat_vs_num.append(feature)

    # === Numeric vs Numeric ===
    if num_vs_num:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Numeric Feature vs Numeric Target</h1>"))
        _spacer_div(20)
        for feature in num_vs_num:
            _plot_numeric_vs_numeric(df, feature, target_col, category_col=color_col)
            _spacer_div(25)
        _spacer_div(35)

    # === Categorical vs Numeric ===
    if cat_vs_num:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Categorical Feature vs Numeric Target</h1>"))
        _spacer_div(20)
        for feature in cat_vs_num:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                _plot_categorical_vs_numeric(df, feature, target_col)
            else:
                _plot_categorical_vs_numeric(df, target_col, feature)
            _spacer_div(25)
        _spacer_div(35)

    # === Categorical vs Categorical ===
    if cat_vs_cat:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Categorical Feature vs Categorical Target</h1>"))
        _spacer_div(20)
        for feature in cat_vs_cat:
            _plot_categorical_vs_categorical(df, feature, target_col)
            _spacer_div(25)
        _spacer_div(35)



# feature_print_count = {}

# def univariate_analysis_category(col_name, cat_columns):
#     global feature_print_count
#     series = cat_columns[col_name]
#     value_counts = series.value_counts(dropna=False)
#     percentage = (value_counts / value_counts.sum()) * 100
#     num_unique = value_counts.shape[0]

#     total_rows = len(series)
#     missing_count = series.isna().sum()
#     missing_pct = (missing_count / total_rows) * 100
#     most_cat = value_counts.idxmax()
#     least_cat = value_counts.idxmin()
#     imbalance_ratio = value_counts.max() / max(value_counts.min(), 1)
#     ent_val = entropy(value_counts)
#     p = value_counts / value_counts.sum()
#     gini_index = 1 - (p**2).sum()
#     mode_ratio = value_counts.max() / total_rows
#     top_n = 1 if num_unique <= 5 else (3 if num_unique <= 20 else 5)

#     feature_print_count[col_name] = feature_print_count.get(col_name, 0) + 1
#     font_size = 24 + 4 * feature_print_count[col_name]

#     display(HTML(f"<div style='margin-bottom:50px;'>"))
#     display(HTML(f"<h2 style='text-align:center; font-size:{font_size}px; color:green;'><b>{col_name}</b></h2>"))

#     print(f"{'Total Rows':<30}: {total_rows}")
#     print(f"{'Unique Categories':<30}: {num_unique}")
#     print(f"{'Missing Values':<30}: {missing_count} ({missing_pct:.2f}%)")
#     print(f"{'Most Frequent Category':<30}: {most_cat} ({value_counts.max()} | {percentage[most_cat]:.2f}%)")
#     print(f"{'Least Frequent Category':<30}: {least_cat} ({value_counts.min()} | {percentage[least_cat]:.2f}%)")
#     print(f"{'Imbalance Ratio (max/min)':<30}: {imbalance_ratio:.2f}")
#     print(f"{'Entropy':<30}: {ent_val:.2f}")
#     print(f"{'Gini Index':<30}: {gini_index:.2f}")
#     print(f"{'Mode Frequency Ratio':<30}: {mode_ratio:.2f}")

#     top_html = "<ul>" + "".join(
#         [f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>" for i, (k, v) in enumerate(value_counts.head(top_n).items())]) + "</ul>"

#     bottom_html = "<ul>" + "".join(
#         [f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>" for i, (k, v) in enumerate(value_counts.tail(top_n).items())]) + "</ul>"

#     table_html = f"""
#     <table style='width:100%; font-size:14px; table-layout:fixed; margin-top:10px;'>
#       <tr>
#         <th style='text-align:center; width:50%; color:#00CED1;'>Top Categories</th>
#         <th style='text-align:center; width:50%; color:#FF69B4;'>Bottom Categories</th>
#       </tr>
#       <tr>
#         <td style='vertical-align:top; padding: 10px;'>{top_html}</td>
#         <td style='vertical-align:top; padding: 10px;'>{bottom_html}</td>
#       </tr>
#     </table>
#     """
#     display(HTML(table_html))

#     colors = ['#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',  
#               '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',  
#               '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB', 
#               '#32CD32', '#00CED1', '#1E90FF', "#FBFB14", '#7CFC00']

#     rotate_labels = -45 if len(value_counts) > 8 else 0

#     fig = make_subplots(
#         rows=1, cols=2, 
#         subplot_titles=(f"{col_name} - Count", f"{col_name} - % Share"),
#         specs=[[{"type": "bar"}, {"type": "domain"}]]
#     )

#     fig.add_trace(
#         go.Bar(
#             x=value_counts.index.astype(str),
#             y=value_counts.values,
#             text=value_counts.values,
#             textposition='outside',
#             marker_color=colors[:len(value_counts)],
#             name=f'{col_name} Count',
#             hovertemplate=f'{col_name}: %{{x}}<br>Count: %{{y}}<extra></extra>'
#         ),
#         row=1, col=1
#     )

#     fig.add_trace(
#         go.Pie(
#             labels=value_counts.index.astype(str),
#             values=percentage,
#             hole=0.45,
#             marker_colors=colors[:len(value_counts)],
#             name=f'{col_name} % Share',
#             hovertemplate=f'{col_name}: %{{label}}<br>Share: %{{percent}}<extra></extra>'
#         ),
#         row=1, col=2
#     )

#     fig.update_layout(
#         title_text=f"Distribution of {col_name}",
#         title_font=dict(size=20, family='Arial'),
#         showlegend=False,
#         plot_bgcolor='#000000',
#         paper_bgcolor='#000000',
#         font=dict(color='white', size=11),
#         height=300,
#         width=670,
#         margin=dict(t=50, b=30, l=30, r=30),
#         xaxis_tickangle=rotate_labels
#     )

#     fig.show()
#     display(HTML("</div>"))





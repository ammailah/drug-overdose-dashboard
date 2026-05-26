import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Setup ──────────────────────────────────────────────────
st.set_page_config(page_title="Drug Overdose Dashboard", layout="wide")
st.title("💊 US Drug Overdose Deaths Dashboard")
st.caption("CDC VSRR Provisional Drug Overdose Death Counts | EDA 350")
st.markdown("---")

# ── Load Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/VSRR_Provisional_Drug_Overdose_Death_Counts.csv")
    df['Data Value'] = pd.to_numeric(df['Data Value'], errors='coerce')
    return df

df = load_data()

# ── Sidebar Filters ─────────────────────────────────────────────
st.sidebar.header("Filters")

# 1. Year
years = sorted(df['Year'].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("Year", years, default=years)

# 2. State
states = sorted(df['State Name'].dropna().unique().tolist())
sel_states = st.sidebar.multiselect("State", states, default=states)

# 3. Indicator / Drug type
indicators = sorted(df['Indicator'].dropna().unique().tolist())
sel_ind = st.sidebar.multiselect("Indicator (Drug Type)", indicators, default=indicators)

# 4. Death count range slider
max_val = int(df['Data Value'].dropna().max())
val_range = st.sidebar.slider("Death Count Range", 0, max_val, (0, max_val))

# 5. Text search
search = st.sidebar.text_input("Search (type any keyword)")

# 6. Reset button
if st.sidebar.button("Reset Filters"):
    st.rerun()

# ── Apply Filters ───────────────────────────────────────────────
fdf = df.copy()
if sel_years:
    fdf = fdf[fdf['Year'].isin(sel_years)]
if sel_states:
    fdf = fdf[fdf['State Name'].isin(sel_states)]
if sel_ind:
    fdf = fdf[fdf['Indicator'].isin(sel_ind)]
fdf = fdf[(fdf['Data Value'] >= val_range[0]) & (fdf['Data Value'] <= val_range[1])]
if search:
    fdf = fdf[fdf.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)]

# ── KPI Cards ───────────────────────────────────────────────────
st.subheader("Key Metrics")
c1, c2, c3, c4 = st.columns(4)
overdose = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths']['Data Value']
c1.metric("Total Records", f"{len(fdf):,}")
c2.metric("Total Deaths", f"{int(overdose.sum()):,}")
c3.metric("Avg Deaths / Record", f"{overdose.mean():.1f}" if len(overdose) > 0 else "N/A")
c4.metric("Max in One Record", f"{int(overdose.max()):,}" if len(overdose) > 0 else "N/A")
st.markdown("---")

# ── Chart helper ────────────────────────────────────────────────
def new_fig():
    fig, ax = plt.subplots(figsize=(8, 4))
    return fig, ax

# ── Row 1: Line + Area ──────────────────────────────────────────
st.subheader("Trends Over Time")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Line Chart – Total Deaths per Year**")
    fig, ax = new_fig()
    data = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths'].groupby('Year')['Data Value'].sum()
    ax.plot(data.index, data.values, marker='o', color='steelblue', linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Deaths"); ax.set_title("Total Overdose Deaths by Year")
    st.pyplot(fig); plt.close()

with col2:
    st.markdown("**Area Chart – Drug Types Over Time**")
    fig, ax = new_fig()
    drugs = ['Cocaine (T40.5)', 'Heroin (T40.1)', 'Methadone (T40.3)']
    area_data = fdf[fdf['Indicator'].isin(drugs)].groupby(['Year','Indicator'])['Data Value'].sum().unstack(fill_value=0)
    area_data.plot.area(ax=ax, alpha=0.7, colormap='tab10')
    ax.set_xlabel("Year"); ax.set_ylabel("Deaths"); ax.set_title("Deaths by Drug Type Over Time")
    ax.legend(fontsize=7, loc='upper left')
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── Row 2: Histogram + Box + Violin ────────────────────────────
st.subheader("Distributions")
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("**Histogram**")
    fig, ax = new_fig()
    vals = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths']['Data Value'].dropna()
    ax.hist(vals, bins=30, color='steelblue', edgecolor='white')
    ax.set_xlabel("Deaths"); ax.set_ylabel("Frequency"); ax.set_title("Death Count Distribution")
    st.pyplot(fig); plt.close()

with col4:
    st.markdown("**Box Plot**")
    fig, ax = new_fig()
    subset = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths'].dropna(subset=['Data Value'])
    years_list = sorted(subset['Year'].unique())
    data_by_year = [subset[subset['Year'] == y]['Data Value'].values for y in years_list]
    ax.boxplot(data_by_year, labels=years_list)
    ax.set_xlabel("Year"); ax.set_ylabel("Deaths"); ax.set_title("Deaths Distribution by Year")
    plt.xticks(rotation=45)
    st.pyplot(fig); plt.close()

with col5:
    st.markdown("**Violin Plot**")
    fig, ax = new_fig()
    subset = fdf[(fdf['Indicator'] == 'Number of Drug Overdose Deaths') & (fdf['Year'] >= 2019)].dropna(subset=['Data Value'])
    sns.violinplot(data=subset, x='Year', y='Data Value', ax=ax, palette='muted')
    ax.set_title("Monthly Deaths Distribution (2019+)")
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── Row 3: Bar + Pie ────────────────────────────────────────────
st.subheader("Comparisons & Composition")
col6, col7 = st.columns(2)

with col6:
    st.markdown("**Bar Chart – Top 15 States**")
    fig, ax = new_fig()
    top = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths'].groupby('State Name')['Data Value'].sum().nlargest(15)
    ax.barh(top.index, top.values, color='steelblue')
    ax.set_xlabel("Total Deaths"); ax.set_title("Top 15 States by Overdose Deaths")
    ax.invert_yaxis()
    st.pyplot(fig); plt.close()

with col7:
    st.markdown("**Pie Chart – Drug Type Share**")
    fig, ax = new_fig()
    pie_drugs = ['Cocaine (T40.5)', 'Heroin (T40.1)', 'Methadone (T40.3)',
                 'Natural & semi-synthetic opioids (T40.2)']
    pie_data = fdf[fdf['Indicator'].isin(pie_drugs)].groupby('Indicator')['Data Value'].sum().dropna()
    ax.pie(pie_data, labels=None, autopct='%1.1f%%', colors=sns.color_palette("Set2", len(pie_data)))
    ax.legend(pie_data.index, loc='lower left', fontsize=7)
    ax.set_title("Drug Type Proportions")
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── Row 4: Heatmap + Scatter + Count ───────────────────────────
st.subheader("Relationships & Counts")
col8, col9, col10 = st.columns(3)

with col8:
    st.markdown("**Heatmap – State × Year**")
    fig, ax = plt.subplots(figsize=(6, 6))
    pivot = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths'].groupby(['State Name','Year'])['Data Value'].sum().unstack(fill_value=0)
    top20 = pivot.sum(axis=1).nlargest(15).index
    sns.heatmap(pivot.loc[top20], cmap='YlOrRd', ax=ax, linewidths=0.3, cbar_kws={'label':'Deaths'})
    ax.set_title("Deaths by State & Year (Top 15)")
    st.pyplot(fig); plt.close()

with col9:
    st.markdown("**Scatter Plot – Deaths vs Year**")
    fig, ax = new_fig()
    sc = fdf[fdf['Indicator'] == 'Number of Drug Overdose Deaths'].groupby('Year')['Data Value'].sum().reset_index()
    ax.scatter(sc['Year'], sc['Data Value'], color='steelblue', s=80, edgecolors='white')
    ax.set_xlabel("Year"); ax.set_ylabel("Total Deaths"); ax.set_title("Total Deaths vs Year")
    st.pyplot(fig); plt.close()

with col10:
    st.markdown("**Count Plot – Records per Year**")
    fig, ax = new_fig()
    counts = fdf.dropna(subset=['Data Value'])['Year'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')
    ax.set_xlabel("Year"); ax.set_ylabel("Record Count"); ax.set_title("Valid Records per Year")
    plt.xticks(rotation=45)
    st.pyplot(fig); plt.close()

st.markdown("---")
st.caption("EDA 350 | Instructor: Ali Hassan Sherazi | Submission: 05-June-2026")

"""
KPI Intelligence Dashboard - Streamlit Web Interface
Optional Bonus Feature
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys

# Import main system components
from main import KPIMonitor, CausalAnalyzer, ActionAgent, ConversationalInterface

# Page configuration
st.set_page_config(
    page_title="KPI Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ffebee;
        padding: 15px;
        border-left: 4px solid #f44336;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        padding: 15px;
        border-left: 4px solid #ff9800;
        border-radius: 5px;
        margin: 10px 0;
    }
    .recommendation {
        background-color: #e8f5e9;
        padding: 15px;
        border-left: 4px solid #4caf50;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'monitor' not in st.session_state:
    try:
        st.session_state.monitor = KPIMonitor('kpi_data.csv')
        if st.session_state.monitor.df is None or st.session_state.monitor.df.empty:
            st.error("❌ Failed to load kpi_data.csv. Please ensure the file exists in the same directory as app.py")
            st.stop()
        st.session_state.analyzer = CausalAnalyzer(st.session_state.monitor.df)
        st.session_state.action_agent = ActionAgent()
        st.session_state.interface = ConversationalInterface(st.session_state.monitor)
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        st.info("Please ensure kpi_data.csv is in the same directory as app.py")
        st.stop()

monitor = st.session_state.monitor
analyzer = st.session_state.analyzer
action_agent = st.session_state.action_agent
interface = st.session_state.interface

# Initialize recommendations in session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# Safety check
if monitor.df is None or monitor.df.empty:
    st.error("❌ No data loaded. Please check kpi_data.csv file.")
    st.stop()

# Title and Header
st.title("📊 KPI Intelligence & Action Agent")
st.markdown("**AI-Powered Business Performance Monitoring System**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Deviation detection settings
    st.subheader("Deviation Detection")
    threshold = st.slider("Threshold (%)", 5, 30, 15) / 100
    window = st.slider("Rolling Window (days)", 3, 14, 7)
    
    # Date range filter
    st.subheader("Date Range")
    if monitor.df is not None and not monitor.df.empty:
        date_range = st.date_input(
            "Select Range",
            value=(monitor.df['Date'].min().date(), monitor.df['Date'].max().date()),
            min_value=monitor.df['Date'].min().date(),
            max_value=monitor.df['Date'].max().date()
        )
    else:
        date_range = None
    
    # Category filter
    st.subheader("Filters")
    categories = ['All'] + list(monitor.df['Category'].unique())
    selected_category = st.selectbox("Category", categories)
    
    st.markdown("---")
    
    # Export buttons
    if st.button("📥 Export Alerts", use_container_width=True):
        if monitor.alerts:
            alerts_df = pd.DataFrame(monitor.alerts)
            alerts_df.to_csv('alerts_log.csv', index=False)
            st.success(f"✓ Exported {len(alerts_df)} alerts to alerts_log.csv")
        else:
            st.info("No alerts to export. Run detection first.")
    
    if st.button("📥 Export Recommendations", use_container_width=True):
        if st.session_state.recommendations:
            rec_df = pd.DataFrame(st.session_state.recommendations)
            rec_df.to_csv('recommendations.csv', index=False)
            st.success(f"✓ Exported {len(rec_df)} recommendations to recommendations.csv")
        else:
            st.warning("⚠️ No recommendations available. Generate them from the 'Causal Analysis' tab first.")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Deviation Detection", "🧪 Causal Analysis", "💬 Chat"])

# TAB 1: Dashboard
with tab1:
    st.header("KPI Performance Dashboard")
    
    # Apply filters
    df_filtered = monitor.df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['Date'] >= pd.to_datetime(date_range[0])) &
            (df_filtered['Date'] <= pd.to_datetime(date_range[1]))
        ]
    if selected_category != 'All':
        df_filtered = df_filtered[df_filtered['Category'] == selected_category]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df_filtered['Revenue_earned'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_sales = df_filtered['Sales_data'].sum()
        st.metric("Total Sales", f"{total_sales:,.0f}")
    
    with col3:
        avg_rating = df_filtered['Rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}/5.0")
    
    with col4:
        avg_discount = df_filtered['Discount'].mean()
        st.metric("Avg Discount", f"{avg_discount:.1f}%")
    
    st.markdown("---")
    
    # Revenue trend chart
    st.subheader("Revenue Trend Over Time")
    daily_revenue = df_filtered.groupby('Date').agg({
        'Revenue_earned': 'sum',
        'Sales_data': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_revenue['Date'],
        y=daily_revenue['Revenue_earned'],
        mode='lines+markers',
        name='Daily Revenue',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Add rolling average
    daily_revenue['rolling_avg'] = daily_revenue['Revenue_earned'].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=daily_revenue['Date'],
        y=daily_revenue['rolling_avg'],
        mode='lines',
        name=f'{window}-Day Rolling Avg',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        category_revenue = df_filtered.groupby('Category')['Revenue_earned'].sum().sort_values(ascending=False)
        fig_cat = px.bar(
            x=category_revenue.values,
            y=category_revenue.index,
            orientation='h',
            labels={'x': 'Revenue ($)', 'y': 'Category'}
        )
        fig_cat.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("Key Metrics Distribution")
        metrics_df = df_filtered[['Supply_Chain_Score', 'Market_Trend_Score', 'Seasonality_Score']].mean()
        fig_metrics = px.bar(
            x=metrics_df.values,
            y=metrics_df.index,
            orientation='h',
            labels={'x': 'Average Score', 'y': 'Metric'}
        )
        fig_metrics.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_metrics, use_container_width=True)

# TAB 2: Deviation Detection
with tab2:
    st.header("🔍 Deviation Detection & Alerts")
    
    if st.button("🚨 Run Deviation Detection", type="primary"):
        with st.spinner("Analyzing KPI deviations..."):
            deviations = monitor.detect_deviations(window=window, threshold=threshold)
            
            if not deviations:
                st.success("✅ No significant deviations detected in the current period")
            else:
                st.warning(f"⚠️ Found {len(deviations)} significant deviation(s)")
                
                for i, dev in enumerate(deviations, 1):
                    alert_type = "alert-critical" if abs(dev['deviation_pct']) > 20 else "alert-warning"
                    
                    st.markdown(f"""
                    <div class="{alert_type}">
                        <h4>Alert #{i}: {dev['type'].upper()} on {dev['date'].strftime('%Y-%m-%d')}</h4>
                        <p><strong>Deviation:</strong> {dev['deviation_pct']:.1f}%</p>
                        <p><strong>Current Revenue:</strong> ${dev['current_revenue']:,.2f}</p>
                        <p><strong>Expected Revenue:</strong> ${dev['baseline_revenue']:,.2f}</p>
                        <p><strong>Impact:</strong> ${abs(dev['current_revenue'] - dev['baseline_revenue']):,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick analysis button
                    if st.button(f"🔬 Analyze Alert #{i}", key=f"analyze_{i}"):
                        st.session_state.selected_deviation = dev
                        st.info("→ Switch to 'Causal Analysis' tab to see detailed analysis")
    
    st.markdown("---")
    
    # Historical alerts
    if monitor.alerts:
        st.subheader("📋 Historical Alerts Summary")
        alerts_df = pd.DataFrame(monitor.alerts)
        alerts_df['date'] = pd.to_datetime(alerts_df['date']).dt.strftime('%Y-%m-%d')
        alerts_df['deviation_pct'] = alerts_df['deviation_pct'].round(2)
        alerts_df['current_revenue'] = alerts_df['current_revenue'].round(2)
        alerts_df['baseline_revenue'] = alerts_df['baseline_revenue'].round(2)
        st.dataframe(alerts_df, use_container_width=True)

# TAB 3: Causal Analysis
with tab3:
    st.header("🧪 Causal Analysis & Recommendations")
    
    # Date selector
    analysis_date = st.date_input(
        "Select Date for Analysis",
        value=monitor.df['Date'].max().date(),
        min_value=monitor.df['Date'].min().date(),
        max_value=monitor.df['Date'].max().date()
    )
    
    if st.button("🔬 Perform Causal Analysis", type="primary"):
        with st.spinner("Analyzing root causes..."):
            target_date = pd.to_datetime(analysis_date)
            
            # Perform causal analysis
            causal = analyzer.analyze_deviation(target_date, window=window)
            
            # Get deviation info
            daily_revenue = monitor.df.groupby('Date')['Revenue_earned'].sum()
            current_rev = daily_revenue[target_date] if target_date in daily_revenue.index else 0
            baseline_rev = daily_revenue.rolling(window).mean()[target_date] if target_date in daily_revenue.index else 0
            
            deviation = {
                'date': target_date,
                'current_revenue': current_rev,
                'baseline_revenue': baseline_rev,
                'deviation_pct': ((current_rev - baseline_rev) / baseline_rev * 100) if baseline_rev > 0 else 0,
                'type': 'drop' if current_rev < baseline_rev else 'spike'
            }
            
            # Display causal factors
            st.subheader("🎯 Root Cause Analysis")
            
            if causal['causes']:
                for i, cause in enumerate(causal['causes'], 1):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {cause['metric']}**")
                    with col2:
                        change_color = "red" if cause['change_pct'] < 0 else "green"
                        st.markdown(f"<span style='color:{change_color}; font-weight:bold;'>{cause['change_pct']:+.1f}%</span>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"`{cause['baseline_value']:.2f} → {cause['current_value']:.2f}`")
                    
                    st.progress(min(abs(cause['change_pct']) / 50, 1.0))
            else:
                st.info("No significant changes detected in key metrics")
            
            st.markdown("---")
            
            # Category impact
            if causal['category_impact']:
                st.subheader("📊 Category-Level Impact")
                
                impact_df = pd.DataFrame(causal['category_impact'])
                fig = px.bar(
                    impact_df,
                    x='change_pct',
                    y='category',
                    orientation='h',
                    labels={'change_pct': 'Revenue Change (%)', 'category': 'Category'},
                    color='change_pct',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Generate recommendations
            st.subheader("💡 Recommended Actions")
            
            recommendations = action_agent.generate_recommendations(causal, deviation['type'])
            
            # IMPORTANT: Save to session state for export
            st.session_state.recommendations = [{
                'date': target_date,
                'deviation_type': deviation['type'],
                'deviation_pct': deviation['deviation_pct'],
                'priority': rec['priority'],
                'cause': rec['cause'],
                'change': rec['change'],
                'action': rec['action'],
                'category': rec['category']
            } for rec in recommendations]
            
            for rec in recommendations:
                priority_emoji = "🔴" if rec['category'] == 'critical' else "🟡"
                
                st.markdown(f"""
                <div class="recommendation">
                    <h4>{priority_emoji} Priority {rec['priority']}: {rec['cause']}</h4>
                    <p><strong>Change:</strong> {rec['change']}</p>
                    <p><strong>Action:</strong> {rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Email report
            st.markdown("---")
            st.subheader("📧 Email-Ready Report")
            
            email_report = action_agent.format_email_report(deviation, causal, recommendations)
            st.text_area("Report Content", email_report, height=400)
            
            # Export button (inline)
            if st.button("📥 Export Recommendations to CSV"):
                rec_df = pd.DataFrame(st.session_state.recommendations)
                rec_df.to_csv('recommendations.csv', index=False)
                st.success("✓ Exported recommendations to recommendations.csv")

# TAB 4: Chat Interface
with tab4:
    st.header("💬 Conversational KPI Query")
    
    st.markdown("""
    Ask natural language questions about your KPIs. Examples:
    - "Show revenue trend for Electronics over last 14 days"
    - "What's the overall summary?"
    - "Show revenue trend for Home & Kitchen last 30 days"
    """)
    
    # Chat input
    user_query = st.text_input("Your question:", placeholder="e.g., Show revenue trend for last 14 days")
    
    if st.button("🔍 Ask", type="primary") and user_query:
        with st.spinner("Processing query..."):
            response = interface.respond(user_query)
            st.text(response)
            
            # Parse query to show relevant visualization
            params = interface.parse_query(user_query)
            
            if params.get('type') == 'trend':
                days = params.get('days', 14)
                category = params.get('category')
                
                trend_df = monitor.get_revenue_trend(
                    category=category,
                    days=days
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_df['Date'],
                    y=trend_df['Revenue_earned'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"Revenue Trend - Last {days} Days",
                    xaxis_title="Date",
                    yaxis_title="Revenue ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Example queries
    st.markdown("---")
    st.subheader("💡 Try these examples:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Overall Summary", use_container_width=True):
            response = interface.respond("Show overall summary")
            st.text(response)
    
    with col2:
        if st.button("📈 Last 7 Days Trend", use_container_width=True):
            response = interface.respond("Show revenue trend for last 7 days")
            st.text(response)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>KPI Intelligence & Action Agent System | Built for ThinkDatax Assignment</p>
</div>
""", unsafe_allow_html=True)
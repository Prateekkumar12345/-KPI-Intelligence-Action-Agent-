"""
KPI Intelligence & Action Agent System
AI Engineer Assignment - ThinkDatax
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import re


class KPIMonitor:
    """Handles KPI data ingestion, tracking, and deviation detection"""
    
    def __init__(self, data_path: str = 'kpi_data.csv'):
        self.df = None
        self.alerts = []
        self.load_data(data_path)
        
    def load_data(self, path: str):
        """Load and prepare KPI dataset"""
        try:
            self.df = pd.read_csv(path)
            column_mapping = {
                'Revenue_d': 'Revenue_earned',
                'Sales_d': 'Sales_data',
                'M_Spend': 'Marketing_Spend',
                'Supply_Chain_E': 'Supply_Chain_Score',
                'Market_T': 'Market_Trend_Score',
                'Seasonality_T': 'Seasonality_Score'
            }
            self.df.rename(columns=column_mapping, inplace=True)
            
            # NOW: Validate required columns
            required_columns = ['Date', 'Revenue_earned']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            
            if missing_cols:
                print(f"✗ Error: Missing required columns: {missing_cols}")
                print(f"✗ Available columns: {list(self.df.columns)}")
                self.df = None
                return
            
            # Convert Date column - handle various date formats
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                # Remove rows with invalid dates
                self.df = self.df.dropna(subset=['Date'])
                self.df = self.df.sort_values('Date')
                
            if len(self.df) == 0:
                print(f"✗ Error: No valid data found in {path}")
                self.df = None
                return
                
            print(f"✓ Loaded {len(self.df)} records from {path}")
            print(f"✓ Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            
        except FileNotFoundError:
            print(f"✗ Error: File '{path}' not found")
            print(f"✗ Please ensure kpi_data.csv is in the same directory as main.py")
            self.df = None
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            print(f"✗ Please check the data file format")
            self.df = None
        
    def standardize_columns(self):
        """Map actual CSV columns to expected column names"""
        column_mapping = {
            'Revenue_d': 'Revenue_earned',
            'Sales_d': 'Sales_data',
            'M_Spend': 'Marketing_Spend',
            'Supply_Chain_E': 'Supply_Chain_Score',
            'Market_T': 'Market_Trend_Score',
            'Seasonality_T': 'Seasonality_Score'
        }
    
        # Rename columns
        self.df.rename(columns=column_mapping, inplace=True)
        print(f"✓ Standardized column names")
            
    def get_revenue_trend(self, category: Optional[str] = None, 
                         product: Optional[str] = None,
                         days: int = 30) -> pd.DataFrame:
        """Get revenue trend with optional filters"""
        df_filtered = self.df.copy()
        
        if category:
            df_filtered = df_filtered[df_filtered['Category'].str.contains(category, case=False, na=False)]
        if product:
            df_filtered = df_filtered[df_filtered['Product_Name'].str.contains(product, case=False, na=False)]
            
        # Get last N days
        max_date = df_filtered['Date'].max()
        min_date = max_date - timedelta(days=days)
        df_filtered = df_filtered[df_filtered['Date'] >= min_date]
        
        # Aggregate by date
        daily_revenue = df_filtered.groupby('Date').agg({
            'Revenue_earned': 'sum',
            'Sales_data': 'sum',
            'Discount': 'mean',
            'Rating': 'mean',
            'Supply_Chain_Score': 'mean',
            'Market_Trend_Score': 'mean'
        }).reset_index()
        
        return daily_revenue
    
    def detect_deviations(self, window: int = 7, threshold: float = 0.15) -> List[Dict]:
        """Detect significant KPI deviations using rolling average"""
        daily_revenue = self.df.groupby('Date')['Revenue_earned'].sum().reset_index()
        
        # Calculate rolling average
        daily_revenue['rolling_avg'] = daily_revenue['Revenue_earned'].rolling(window=window, min_periods=1).mean()
        daily_revenue['deviation_pct'] = (daily_revenue['Revenue_earned'] - daily_revenue['rolling_avg']) / daily_revenue['rolling_avg']
        
        # Flag deviations
        deviations = []
        for idx, row in daily_revenue.iterrows():
            if abs(row['deviation_pct']) > threshold and idx >= window:
                deviation = {
                    'date': row['Date'],
                    'current_revenue': row['Revenue_earned'],
                    'baseline_revenue': row['rolling_avg'],
                    'deviation_pct': row['deviation_pct'] * 100,
                    'type': 'drop' if row['deviation_pct'] < 0 else 'spike'
                }
                deviations.append(deviation)
                self.alerts.append(deviation)
        
        return deviations
    
    def get_kpi_summary(self, date_range: Optional[Tuple[str, str]] = None) -> Dict:
        """Get overall KPI summary"""
        df_filtered = self.df.copy()
        
        if date_range:
            start, end = date_range
            df_filtered = df_filtered[(df_filtered['Date'] >= start) & (df_filtered['Date'] <= end)]
        
        summary = {
            'total_revenue': df_filtered['Revenue_earned'].sum(),
            'total_sales': df_filtered['Sales_data'].sum(),
            'avg_discount': df_filtered['Discount'].mean(),
            'avg_rating': df_filtered['Rating'].mean(),
            'avg_supply_chain': df_filtered['Supply_Chain_Score'].mean(),
            'avg_market_trend': df_filtered['Market_Trend_Score'].mean(),
            'record_count': len(df_filtered)
        }
        
        return summary


class CausalAnalyzer:
    """Performs causal analysis on KPI deviations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_deviation(self, deviation_date: datetime, window: int = 7) -> Dict:
        """Analyze causes of deviation"""
        # Get baseline period (before deviation)
        baseline_end = deviation_date - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=window)
        
        # Get deviation period
        deviation_start = deviation_date
        deviation_end = deviation_date + timedelta(days=window-1)
        
        baseline = self.df[(self.df['Date'] >= baseline_start) & (self.df['Date'] <= baseline_end)]
        deviation_period = self.df[(self.df['Date'] >= deviation_start) & (self.df['Date'] <= deviation_end)]
        
        # Calculate changes in key metrics
        causes = []
        
        # Revenue components
        metrics = {
            'Sales_data': 'Monthly Sales',
            'Discount': 'Discount Rate',
            'Rating': 'Product Rating',
            'Supply_Chain_Score': 'Supply Chain Efficiency',
            'Market_Trend_Score': 'Market Trend Score',
            'Marketing_Spend': 'Marketing Spend',
            'Seasonality_Score': 'Seasonality Score',
            'Success_Percentage': 'Success Rate'
        }
        
        for col, label in metrics.items():
            if col in baseline.columns and col in deviation_period.columns:
                baseline_val = baseline[col].mean()
                deviation_val = deviation_period[col].mean()
                
                if baseline_val > 0:
                    pct_change = ((deviation_val - baseline_val) / baseline_val) * 100
                    
                    if abs(pct_change) > 5:  # Significant change threshold
                        causes.append({
                            'metric': label,
                            'baseline_value': round(baseline_val, 2),
                            'current_value': round(deviation_val, 2),
                            'change_pct': round(pct_change, 2),
                            'impact': 'high' if abs(pct_change) > 15 else 'medium'
                        })
        
        # Sort by absolute change
        causes.sort(key=lambda x: abs(x['change_pct']), reverse=True)
        
        # Category-level analysis
        category_impact = self._analyze_category_impact(baseline, deviation_period)
        
        return {
            'causes': causes[:5],  # Top 5 causes
            'category_impact': category_impact
        }
    
    def _analyze_category_impact(self, baseline: pd.DataFrame, deviation: pd.DataFrame) -> List[Dict]:
        """Analyze impact by category"""
        baseline_cat = baseline.groupby('Category')['Revenue_earned'].sum()
        deviation_cat = deviation.groupby('Category')['Revenue_earned'].sum()
        
        impact = []
        for cat in baseline_cat.index:
            if cat in deviation_cat.index:
                base_rev = baseline_cat[cat]
                dev_rev = deviation_cat[cat]
                pct_change = ((dev_rev - base_rev) / base_rev) * 100
                
                impact.append({
                    'category': cat,
                    'baseline_revenue': round(base_rev, 2),
                    'current_revenue': round(dev_rev, 2),
                    'change_pct': round(pct_change, 2)
                })
        
        impact.sort(key=lambda x: abs(x['change_pct']), reverse=True)
        return impact[:3]


class ActionAgent:
    """Generates intelligent action recommendations"""
    
    def __init__(self):
        self.action_templates = {
            'Sales_data': {
                'decrease': [
                    "Increase marketing spend by {intensity}% for affected categories",
                    "Launch promotional campaigns to boost sales volume",
                    "Review and optimize product listings and descriptions"
                ],
                'increase': [
                    "Scale successful marketing campaigns",
                    "Invest in high-performing product categories"
                ]
            },
            'Discount': {
                'decrease': [
                    "Adjust discount strategy to regain competitive positioning",
                    "Implement targeted promotional offers for price-sensitive segments"
                ],
                'increase': [
                    "Review discount sustainability and margin impact",
                    "Optimize discount allocation to maintain profitability"
                ]
            },
            'Rating': {
                'decrease': [
                    "Investigate product quality issues and customer feedback",
                    "Enhance customer service and support response times",
                    "Review recent product changes or supplier quality"
                ],
                'increase': [
                    "Leverage positive reviews in marketing materials",
                    "Expand successful product lines"
                ]
            },
            'Supply_Chain_Score': {
                'decrease': [
                    "Review supplier performance and delivery timelines",
                    "Implement inventory buffer for critical products",
                    "Investigate logistics bottlenecks and fulfillment delays"
                ],
                'increase': [
                    "Document and replicate supply chain best practices",
                    "Consider expanding capacity for high-demand items"
                ]
            },
            'Market_Trend_Score': {
                'decrease': [
                    "Conduct market research to identify shifting consumer preferences",
                    "Adjust product mix to align with current market trends",
                    "Monitor competitor activities and pricing strategies"
                ],
                'increase': [
                    "Capitalize on positive market momentum with increased inventory",
                    "Accelerate product launches in trending categories"
                ]
            },
            'Marketing_Spend': {
                'decrease': [
                    "Reassess marketing budget allocation and ROI",
                    "Shift to higher-performing marketing channels"
                ],
                'increase': [
                    "Monitor marketing efficiency and cost per acquisition",
                    "Ensure marketing spend is translating to revenue growth"
                ]
            }
        }
    
    def generate_recommendations(self, causal_analysis: Dict, deviation_type: str) -> List[Dict]:
        """Generate prioritized action recommendations"""
        recommendations = []
        causes = causal_analysis['causes']
        
        for idx, cause in enumerate(causes[:3], 1):  # Top 3 causes
            metric = cause['metric']
            change_pct = cause['change_pct']
            
            # Determine direction and intensity
            direction = 'decrease' if change_pct < 0 else 'increase'
            intensity = 20 if abs(change_pct) > 20 else 15 if abs(change_pct) > 10 else 10
            
            # Map metric to action template
            for key, templates in self.action_templates.items():
                if key.replace('_', ' ').lower() in metric.lower():
                    actions = templates.get(direction, [])
                    if actions:
                        action_text = actions[0].format(intensity=intensity)
                        recommendations.append({
                            'priority': idx,
                            'cause': metric,
                            'change': f"{change_pct:+.1f}%",
                            'action': action_text,
                            'category': 'critical' if abs(change_pct) > 20 else 'important'
                        })
                        break
        
        # Add category-specific recommendations
        if causal_analysis['category_impact']:
            top_category = causal_analysis['category_impact'][0]
            if abs(top_category['change_pct']) > 15:
                recommendations.append({
                    'priority': len(recommendations) + 1,
                    'cause': f"{top_category['category']} Performance",
                    'change': f"{top_category['change_pct']:+.1f}%",
                    'action': f"Focus intervention on {top_category['category']} category - primary revenue impact driver",
                    'category': 'critical'
                })
        
        return recommendations
    
    def format_email_report(self, deviation: Dict, causal_analysis: Dict, 
                           recommendations: List[Dict]) -> str:
        """Format recommendations as email-ready report"""
        email = f"""
Subject: {'Revenue Decline' if deviation['type'] == 'drop' else 'Revenue Spike'} Alert – Recommended Actions

Date: {deviation['date'].strftime('%Y-%m-%d')}
Alert Type: {deviation['type'].upper()}
Deviation: {deviation['deviation_pct']:.1f}%
Current Revenue: ${deviation['current_revenue']:,.2f}
Baseline Revenue: ${deviation['baseline_revenue']:,.2f}

CAUSAL ANALYSIS
{'='*50}
"""
        
        for idx, cause in enumerate(causal_analysis['causes'][:3], 1):
            email += f"\n{idx}. {cause['metric']}: {cause['change_pct']:+.1f}% change"
            email += f"\n   Baseline: {cause['baseline_value']:.2f} → Current: {cause['current_value']:.2f}"
        
        email += f"\n\nRECOMMENDED ACTIONS\n{'='*50}\n"
        
        for rec in recommendations:
            email += f"\n[{rec['category'].upper()}] Priority {rec['priority']}"
            email += f"\n• {rec['action']}"
            email += f"\n  Cause: {rec['cause']} ({rec['change']})\n"
        
        return email


class ConversationalInterface:
    """Handles natural language queries about KPIs"""
    
    def __init__(self, monitor: KPIMonitor):
        self.monitor = monitor
        
    def parse_query(self, query: str) -> Dict:
        """Parse natural language query into structured parameters"""
        query_lower = query.lower()
        params = {}
        
        # Extract time period
        if 'last' in query_lower:
            match = re.search(r'last (\d+) days?', query_lower)
            if match:
                params['days'] = int(match.group(1))
        
        # Extract category
        categories = self.monitor.df['Category'].unique()
        for cat in categories:
            if cat.lower() in query_lower:
                params['category'] = cat
                break
        
        # Extract product
        if 'product' in query_lower or any(prod in query_lower for prod in ['fryer', 'yoga', 'blender']):
            products = self.monitor.df['Product_Name'].unique()
            for prod in products:
                if any(word in query_lower for word in prod.lower().split()):
                    params['product'] = prod
                    break
        
        # Determine query type
        if 'trend' in query_lower or 'revenue' in query_lower:
            params['type'] = 'trend'
        elif 'summary' in query_lower or 'overview' in query_lower:
            params['type'] = 'summary'
        elif 'deviation' in query_lower or 'alert' in query_lower:
            params['type'] = 'deviation'
        
        return params
    
    def respond(self, query: str) -> str:
        """Generate response to natural language query"""
        params = self.parse_query(query)
        
        if params.get('type') == 'trend':
            days = params.get('days', 14)
            category = params.get('category')
            product = params.get('product')
            
            trend_df = self.monitor.get_revenue_trend(
                category=category, 
                product=product, 
                days=days
            )
            
            filter_desc = []
            if category:
                filter_desc.append(f"category: {category}")
            if product:
                filter_desc.append(f"product: {product}")
            filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
            
            total_revenue = trend_df['Revenue_earned'].sum()
            avg_daily = trend_df['Revenue_earned'].mean()
            
            response = f"\n{'='*60}\n"
            response += f"REVENUE TREND ANALYSIS{filter_str}\n"
            response += f"Period: Last {days} days\n"
            response += f"{'='*60}\n\n"
            response += f"Total Revenue: ${total_revenue:,.2f}\n"
            response += f"Average Daily Revenue: ${avg_daily:,.2f}\n"
            response += f"Data Points: {len(trend_df)} days\n"
            
            if len(trend_df) > 1:
                first_week_avg = trend_df.head(min(7, len(trend_df)))['Revenue_earned'].mean()
                last_week_avg = trend_df.tail(min(7, len(trend_df)))['Revenue_earned'].mean()
                pct_change = ((last_week_avg - first_week_avg) / first_week_avg) * 100
                
                response += f"\nTrend: {pct_change:+.1f}% "
                response += f"({'Growing' if pct_change > 0 else 'Declining'})\n"
            
            return response
            
        elif params.get('type') == 'summary':
            summary = self.monitor.get_kpi_summary()
            
            response = f"\n{'='*60}\n"
            response += "KPI PERFORMANCE SUMMARY\n"
            response += f"{'='*60}\n\n"
            response += f"Total Revenue: ${summary['total_revenue']:,.2f}\n"
            response += f"Total Sales: {summary['total_sales']:,.0f} units\n"
            response += f"Average Discount: {summary['avg_discount']:.1f}%\n"
            response += f"Average Rating: {summary['avg_rating']:.2f}/5.0\n"
            response += f"Supply Chain Score: {summary['avg_supply_chain']:.1f}\n"
            response += f"Market Trend Score: {summary['avg_market_trend']:.1f}\n"
            
            return response
        
        else:
            return "\nI can help you with:\n• Revenue trends (e.g., 'Show revenue trend for last 14 days')\n• KPI summaries (e.g., 'Show overall summary')\n• Deviation analysis (use 'detect' command)\n"


class KPIIntelligenceSystem:
    """Main system orchestrator"""
    
    def __init__(self, data_path: str = 'kpi_data.csv'):
        print("\n" + "="*60)
        print("KPI INTELLIGENCE & ACTION AGENT SYSTEM")
        print("="*60)
        
        self.monitor = KPIMonitor(data_path)
        
        # Check if data loaded successfully
        if self.monitor.df is None:
            print("\n✗ FATAL ERROR: Could not load data file")
            print("✗ Please ensure kpi_data.csv exists in the current directory")
            print("✗ System cannot start without data")
            import sys
            sys.exit(1)
        
        self.analyzer = CausalAnalyzer(self.monitor.df)
        self.action_agent = ActionAgent()
        self.interface = ConversationalInterface(self.monitor)
        
    def run_cli(self):
        """Run command-line interface"""
        print("\n✓ System initialized successfully")
        print("\nAvailable commands:")
        print("  - chat: Conversational query mode")
        print("  - detect: Run deviation detection")
        print("  - analyze <date>: Analyze specific date (YYYY-MM-DD)")
        print("  - export: Export alerts and recommendations")
        print("  - quit: Exit system\n")
        
        while True:
            try:
                cmd = input(">>> ").strip()
                
                if not cmd:
                    continue
                    
                if cmd.lower() == 'quit':
                    print("\nShutting down system...")
                    break
                    
                elif cmd.lower() == 'chat':
                    self.chat_mode()
                    
                elif cmd.lower() == 'detect':
                    self.run_detection()
                    
                elif cmd.lower().startswith('analyze'):
                    parts = cmd.split()
                    if len(parts) > 1:
                        self.analyze_date(parts[1])
                    else:
                        print("Usage: analyze YYYY-MM-DD")
                        
                elif cmd.lower() == 'export':
                    self.export_results()
                    
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nShutting down system...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def chat_mode(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("CONVERSATIONAL QUERY MODE")
        print("="*60)
        print("Ask questions about your KPIs. Type 'exit' to return.\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if query.lower() == 'exit':
                    break
                    
                if query:
                    response = self.interface.respond(query)
                    print(response)
                    
            except KeyboardInterrupt:
                break
    
    def run_detection(self):
        """Run deviation detection"""
        print("\nRunning deviation detection...")
        deviations = self.monitor.detect_deviations(window=7, threshold=0.15)
        
        if not deviations:
            print("✓ No significant deviations detected")
            return
        
        print(f"\n⚠ Found {len(deviations)} deviation(s):\n")
        
        for dev in deviations:
            print(f"Date: {dev['date'].strftime('%Y-%m-%d')}")
            print(f"Type: {dev['type'].upper()}")
            print(f"Deviation: {dev['deviation_pct']:.1f}%")
            print(f"Current: ${dev['current_revenue']:,.2f}")
            print(f"Baseline: ${dev['baseline_revenue']:,.2f}")
            print("-" * 40)
    
    def analyze_date(self, date_str: str):
        """Analyze specific date"""
        try:
            target_date = pd.to_datetime(date_str)
            
            print(f"\nAnalyzing {date_str}...")
            
            # Get deviation info
            daily_revenue = self.monitor.df.groupby('Date')['Revenue_earned'].sum()
            if target_date not in daily_revenue.index:
                print(f"No data found for {date_str}")
                return
            
            # Perform causal analysis
            causal = self.analyzer.analyze_deviation(target_date, window=7)
            
            # Generate recommendations
            deviation = {
                'date': target_date,
                'current_revenue': daily_revenue[target_date],
                'baseline_revenue': daily_revenue.rolling(7).mean()[target_date],
                'deviation_pct': 0,  # Calculate if needed
                'type': 'drop'
            }
            
            recommendations = self.action_agent.generate_recommendations(causal, 'drop')
            
            # Display results
            print("\n" + "="*60)
            print("CAUSAL ANALYSIS")
            print("="*60)
            
            for cause in causal['causes']:
                print(f"\n• {cause['metric']}: {cause['change_pct']:+.1f}%")
                print(f"  {cause['baseline_value']:.2f} → {cause['current_value']:.2f}")
            
            print("\n" + "="*60)
            print("RECOMMENDED ACTIONS")
            print("="*60)
            
            for rec in recommendations:
                print(f"\n[Priority {rec['priority']}] {rec['action']}")
                print(f"  Addressing: {rec['cause']} ({rec['change']})")
            
        except Exception as e:
            print(f"Error analyzing date: {e}")
    
    def export_results(self):
        """Export alerts and recommendations to CSV"""
        try:
            # Export alerts
            if self.monitor.alerts:
                alerts_df = pd.DataFrame(self.monitor.alerts)
                alerts_df.to_csv('alerts_log.csv', index=False)
                print(f"✓ Exported {len(alerts_df)} alerts to alerts_log.csv")
            
            # Generate and export recommendations for all alerts
            all_recommendations = []
            for alert in self.monitor.alerts:
                causal = self.analyzer.analyze_deviation(alert['date'], window=7)
                recommendations = self.action_agent.generate_recommendations(causal, alert['type'])
                
                for rec in recommendations:
                    all_recommendations.append({
                        'date': alert['date'],
                        'deviation_type': alert['type'],
                        'deviation_pct': alert['deviation_pct'],
                        'priority': rec['priority'],
                        'cause': rec['cause'],
                        'change': rec['change'],
                        'action': rec['action'],
                        'category': rec['category']
                    })
            
            if all_recommendations:
                rec_df = pd.DataFrame(all_recommendations)
                rec_df.to_csv('recommendations.csv', index=False)
                print(f"✓ Exported {len(rec_df)} recommendations to recommendations.csv")
            
        except Exception as e:
            print(f"Error exporting results: {e}")


if __name__ == "__main__":
    system = KPIIntelligenceSystem('kpi_data.csv')
    system.run_cli()
# -KPI-Intelligence-Action-Agent-

The system simulates a business intelligence layer for a retail / ecommerce brand. It 
should continuously track performance and enable stakeholders to understand: 
• What changed in performance? 
• Why did it change? 
• What actions should be taken?

# The 3 Levers 
# Lever 1: KPI Monitoring & Conversational Interface 
The system should: 
• Ingest the provided 90-day dataset 
• Track Overall Revenue and key KPIs on a daily basis 
• Support filtering by dimensions (Product, Category, Date Range, etc.) 
• Allow conversational interaction with data (chatbot-style querying) 
• Flag deviations based on configurable thresholds 
• Display current KPI performance vs baseline

# Lever 2: Causal Analysis Engine (Deviation Intelligence)
When a KPI deviation is detected, the system should: 
• Analyse contributing variables from the dataset 
• Identify statistically or logically significant drivers 
• Produce ranked hypotheses explaining the deviation 
• Present business-readable causal insights 
Possible causes may include: 
• Drop in Sales_m 
• Spike or reduction in Discount 
• Decline in customer Ratings 
 # Lever 3: Intelligent Action Agent (Recommendation Engine)
 Based on discovered causes, the system should: 
• Generate prioritised, actionable recommendations 
• Map each cause to a business intervention 
• Draft action guidance in concise business language 
• Present recommendations in an email-ready format

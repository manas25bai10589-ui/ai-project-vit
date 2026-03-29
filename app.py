import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
from datetime import date, timedelta
import logging

# CONFIGURATION & LOGS
st.set_page_config(page_title="NexTrend Inventory AI", layout="wide", page_icon="📦")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PART 1: USER MANAGEMENT ---
class UserManager:
    def __init__(self):
        # In a real app, hash these passwords!
        self.users = {
            "admin": "admin123",       
            "planner": "supply2025"    
        }

    def login(self, username, password):
        if username in self.users and self.users[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            logger.info(f"User {username} logged in successfully.")
            return True
        return False

    def logout(self):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None

# --- PART 2: DATA INPUT & PROCESSING ---
class DataProcessor:
    """
    Fetches historical data and treats Stock Price as 'Daily Demand'.
    """
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_data(ticker, start_date, end_date):
        try:
            # Fetch data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                return None
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Ensure Date column exists (sometimes yfinance index naming varies)
            if 'Date' not in df.columns:
                 df.rename(columns={'index': 'Date'}, inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    @staticmethod
    def prepare_data_for_model(df):
        # Feature Engineering: Date to Ordinal
        df = df.copy()
        df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
        X = df[['Date_Ordinal']]
        
        # We use 'Close' price as a proxy for 'Daily Sales Demand'
        # We handle cases where 'Close' might be a tuple (multi-index) by squeezing
        if isinstance(df['Close'], pd.DataFrame):
             y = df['Close'].iloc[:, 0]
        else:
             y = df['Close']
             
        return X, y

# --- PART 3: PREDICTION & ANALYTICS ---
class TrendPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train_and_predict(self, X, y, future_days=30):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train Model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Forecast Future
        last_date = X.iloc[-1, 0]
        future_dates_ordinal = np.array([last_date + i for i in range(1, future_days + 1)]).reshape(-1, 1)
        future_demand = self.model.predict(future_dates_ordinal)
        
        return {
            'model': self.model,
            'mse': mse,
            'r2': r2,
            'future_dates_ordinal': future_dates_ordinal,
            'future_demand': future_demand,
            'X_test': X_test,
            'y_test': y_test
        }

# --- PART 4: INVENTORY INTELLIGENCE (NEW CLASS) ---
class InventoryAnalyst:
    """
    Converts pure Demand Data into Supply Chain Actionables.
    """
    @staticmethod
    def calculate_metrics(current_demand, predicted_demand):
        # Simulate Lead Time (Time to get new stock) - normally from DB
        avg_lead_time = 5  # days
        max_lead_time = 8  # days
        
        # 1. Safety Stock = (Max Daily Usage * Max Lead Time) - (Avg Daily Usage * Avg Lead Time)
        # We estimate Max Usage as 1.2x current demand for safety
        max_usage = current_demand * 1.2
        avg_usage = current_demand
        
        safety_stock = (max_usage * max_lead_time) - (avg_usage * avg_lead_time)
        
        # 2. Reorder Point = (Avg Daily Usage * Avg Lead Time) + Safety Stock
        reorder_point = (avg_usage * avg_lead_time) + safety_stock
        
        # 3. Status
        trend = "Rising" if predicted_demand > current_demand else "Falling"
        
        return {
            "safety_stock": max(0, round(safety_stock, 2)),
            "reorder_point": max(0, round(reorder_point, 2)),
            "trend": trend,
            "recommendation": "INCREASE ORDER SIZE" if trend == "Rising" else "MAINTAIN LEVELS"
        }

# --- PART 5: UI ---
class Dashboard:
    def render_login(self, user_manager):
        st.title("NexTrend Analytics")
        st.markdown("### Supply Chain Intelligence Portal")
        st.info("Log in to access sensitive inventory data.")
        
        with st.form("login_form"):
            username = st.text_input("Username (Try: admin)")
            password = st.text_input("Password (Try: admin123)", type="password")
            submitted = st.form_submit_button("Access Dashboard")
            
            if submitted:
                if user_manager.login(username, password):
                    st.success("Authenticated.")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    def render_main(self, user_manager, data_processor, predictor, analyst):
        # Sidebar
        with st.sidebar:
            st.title(f"👤 {st.session_state['username']}")
            st.caption("Role: Supply Chain Planner")
            st.divider()
            
            st.header("⚙️ Configuration")
            ticker = st.text_input("Product Ticker (Demand Proxy)", "AMZN", help="Uses stock volume as proxy for demand")
            years = st.slider("Historical Analysis (Years)", 1, 5, 2)
            future_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
            
            st.divider()
            if st.button("Logout", type="primary"):
                user_manager.logout()
                st.rerun()

        # Main Content
        st.title("📦 Inventory Intelligence Engine")
        st.markdown(f"Analyzing Demand Velocity for: **{ticker}**")
        
        start_date = date.today() - timedelta(days=years*365)
        end_date = date.today()

        with st.spinner('Fetching market data and calculating trends...'):
            data = data_processor.load_data(ticker, start_date, end_date)

        if data is not None and not data.empty:
            # Process & Predict
            X, y = data_processor.prepare_data_for_model(data)
            results = predictor.train_and_predict(X, y, future_days)
            
            # Analysis
            current_demand = float(y.iloc[-1])
            predicted_demand = float(results['future_demand'][-1])
            metrics = analyst.calculate_metrics(current_demand, predicted_demand)
            
            # --- KPI ROW ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Daily Demand", f"{current_demand:.2f} units")
            col2.metric("Projected Demand", f"{predicted_demand:.2f} units", 
                        delta=f"{((predicted_demand-current_demand)/current_demand)*100:.1f}%")
            col3.metric("Rec. Safety Stock", f"{metrics['safety_stock']} units")
            col4.metric("Reorder Point", f"{metrics['reorder_point']} units")

            # --- VISUALIZATION TAB ---
            tab1, tab2 = st.tabs(["📉 Forecast Visualization", "📋 Smart Alerts"])
            
            with tab1:
                # Prepare dates
                future_dt = [date.fromordinal(int(d[0])) for d in results['future_dates_ordinal']]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=y, name="Historical Demand", line=dict(color='#0068C9', width=2)))
                fig.add_trace(go.Scatter(x=future_dt, y=results['future_demand'], name="AI Forecast", line=dict(color='#29B09D', width=3, dash='dash')))
                
                fig.update_layout(
                    title="Demand Velocity Trajectory",
                    xaxis_title="Timeline",
                    yaxis_title="Demand Index (Normalized)",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("📢 Supply Chain Recommendations")
                
                # Logic to display "Smart" alerts
                if metrics['trend'] == "Rising":
                    st.warning(f"⚠️ **High Demand Alert**: Demand is projected to rise. {metrics['recommendation']}")
                else:
                    st.success(f"✅ **Demand Stable**: {metrics['recommendation']}")
                
                st.info(f"""
                **Analysis Logic:**
                - Based on a calculated lead time of 5-8 days.
                - Reorder Point triggered when stock falls below **{metrics['reorder_point']}** units.
                - Safety Stock buffer established at **{metrics['safety_stock']}** units to prevent stockouts.
                """)

        else:
            st.error(f"Unable to fetch data for ticker: {ticker}. Please try a valid symbol (e.g., AAPL, TSLA).")

# --- MAIN APP ENTRY POINT ---
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    user_manager = UserManager()
    data_processor = DataProcessor()
    predictor = TrendPredictor()
    analyst = InventoryAnalyst()
    dashboard = Dashboard()

    if not st.session_state['logged_in']:
        dashboard.render_login(user_manager)
    else:
        dashboard.render_main(user_manager, data_processor, predictor, analyst)

if __name__ == "__main__":
    main()
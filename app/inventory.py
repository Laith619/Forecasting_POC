import pandas as pd
import numpy as np
from typing import Dict, Union, Tuple, Optional
from datetime import datetime, timedelta

class InventoryManager:
    def __init__(
        self,
        lead_time: int = 14,
        service_level: float = 0.95,
        review_period: int = 7
    ):
        """
        Initialize Inventory Manager.
        
        Args:
            lead_time: Lead time in days
            service_level: Service level (e.g., 0.95 for 95%)
            review_period: Review period in days
        """
        self.lead_time = lead_time
        self.service_level = service_level
        self.review_period = review_period
        self.z_score = self._get_z_score(service_level)
        
    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for given service level."""
        # Common z-scores for service levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.645,
            0.98: 2.055,
            0.99: 2.326
        }
        return z_scores.get(service_level, 1.645)  # Default to 95% service level
        
    def calculate_safety_stock(
        self,
        demand_forecast: pd.DataFrame,
        forecast_std: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate safety stock levels.
        
        Args:
            demand_forecast: DataFrame with forecast data
            forecast_std: Standard deviation of forecast (optional)
        """
        if forecast_std is None:
            forecast_std = demand_forecast['yhat'].std()
            
        # Calculate safety stock using the square root of lead time
        safety_stock = self.z_score * forecast_std * np.sqrt(self.lead_time)
        
        return {
            'safety_stock': safety_stock,
            'std_dev': forecast_std
        }
        
    def calculate_reorder_point(
        self,
        demand_forecast: pd.DataFrame,
        safety_stock: float
    ) -> float:
        """
        Calculate reorder point.
        
        Args:
            demand_forecast: DataFrame with forecast data
            safety_stock: Calculated safety stock level
        """
        # Calculate average daily demand
        avg_demand = demand_forecast['yhat'].mean()
        
        # Calculate reorder point
        rop = (avg_demand * self.lead_time) + safety_stock
        
        return rop
        
    def assess_stockout_risk(
        self,
        current_stock: float,
        demand_forecast: pd.DataFrame
    ) -> Dict[str, Union[float, str]]:
        """
        Assess risk of stockout.
        
        Args:
            current_stock: Current inventory level
            demand_forecast: DataFrame with forecast data
        """
        # Calculate cumulative demand
        cumulative_demand = demand_forecast['yhat'].cumsum()
        
        # Find days until stockout
        days_to_stockout = np.searchsorted(cumulative_demand, current_stock)
        
        # Determine risk level
        risk_level = 'LOW'
        if days_to_stockout < self.lead_time:
            risk_level = 'HIGH'
        elif days_to_stockout < self.lead_time * 2:
            risk_level = 'MEDIUM'
            
        return {
            'days_to_stockout': days_to_stockout,
            'risk_level': risk_level
        }
        
    def calculate_order_quantity(
        self,
        demand_forecast: pd.DataFrame,
        current_stock: float,
        min_order: float = 0
    ) -> Dict[str, float]:
        """
        Calculate optimal order quantity.
        
        Args:
            demand_forecast: DataFrame with forecast data
            current_stock: Current inventory level
            min_order: Minimum order quantity
        """
        # Calculate expected demand during lead time + review period
        planning_period = self.lead_time + self.review_period
        expected_demand = demand_forecast['yhat'].head(planning_period).sum()
        
        # Calculate safety stock
        safety_stock_info = self.calculate_safety_stock(demand_forecast)
        safety_stock = safety_stock_info['safety_stock']
        
        # Calculate order quantity
        order_qty = max(
            expected_demand + safety_stock - current_stock,
            min_order
        )
        
        return {
            'order_quantity': order_qty,
            'expected_demand': expected_demand,
            'safety_stock': safety_stock
        }
        
    def get_inventory_metrics(
        self,
        demand_forecast: pd.DataFrame,
        current_stock: float
    ) -> Dict[str, Union[float, str]]:
        """
        Get comprehensive inventory metrics.
        
        Args:
            demand_forecast: DataFrame with forecast data
            current_stock: Current inventory level
        """
        # Calculate all metrics
        safety_stock_info = self.calculate_safety_stock(demand_forecast)
        safety_stock = safety_stock_info['safety_stock']
        
        reorder_point = self.calculate_reorder_point(
            demand_forecast,
            safety_stock
        )
        
        stockout_risk = self.assess_stockout_risk(
            current_stock,
            demand_forecast
        )
        
        order_info = self.calculate_order_quantity(
            demand_forecast,
            current_stock
        )
        
        # Combine all metrics
        metrics = {
            'current_stock': current_stock,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'days_to_stockout': stockout_risk['days_to_stockout'],
            'risk_level': stockout_risk['risk_level'],
            'suggested_order': order_info['order_quantity'],
            'expected_demand': order_info['expected_demand']
        }
        
        return metrics 
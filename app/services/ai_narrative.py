import anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
import logging

logger = logging.getLogger(__name__)

def generate_commentary(
    current_price: float,
    end_price: float,
    lower_80: float,
    upper_80: float,
    lower_95: float,
    upper_95: float,
    mape: float,
    horizon_label: str,
    trend_30d: float,
    trend_90d: float,
    seasonality_mode: str,
) -> str:
    """Generate AI narrative commentary using Claude."""
    if not ANTHROPIC_API_KEY:
        return "AI commentary unavailable (no API key configured)."

    direction = "increase" if end_price > current_price else "decrease"
    pct_change = ((end_price - current_price) / current_price) * 100
    current_lb = current_price / 2204.62
    end_lb = end_price / 2204.62
    lower_80_lb = lower_80 / 2204.62
    upper_80_lb = upper_80 / 2204.62

    prompt = f"""You are a commodity pricing analyst for a galvanizing company that consumes zinc as its primary raw material. Analyze the following zinc price forecast and provide actionable commentary for the procurement team.

CURRENT MARKET DATA:
- Latest LME zinc spot price: ${current_lb:.4f}/lb (${current_price:,.0f}/MT)
- 30-day trend: {trend_30d:+.1f}%
- 90-day trend: {trend_90d:+.1f}%

FORECAST ({horizon_label}):
- Predicted price at horizon end: ${end_lb:.4f}/lb (${end_price:,.0f}/MT)
- Direction: {direction} ({pct_change:+.1f}% from current)
- 80% confidence range: ${lower_80_lb:.4f} - ${upper_80_lb:.4f}/lb (${lower_80:,.0f} - ${upper_80:,.0f}/MT)
- 95% confidence range: ${lower_95/2204.62:.4f} - ${upper_95/2204.62:.4f}/lb (${lower_95:,.0f} - ${upper_95:,.0f}/MT)
- Model: Holt-Winters ({seasonality_mode} seasonality)
- Model accuracy (MAPE): {mape:.1f}%

Provide:
1. A 2-3 sentence executive summary of the forecast
2. A purchasing recommendation (accelerate purchases, maintain current pace, or consider deferring)
3. Key risk factors to monitor
4. Any notable seasonal patterns the model detected

Keep the language clear for procurement and finance professionals, not data scientists. Lead with USD per pound since that is what the galvanizing industry uses. Reference USD per metric ton parenthetically. Do not use em dashes; use commas, periods, or restructure sentences instead."""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"AI narrative error: {e}")
        return f"AI commentary temporarily unavailable: {e}"

import pandas as pd

def ytm(price, coupon, years_to_maturity):
    cash_flow = [100+coupon/2]
    cash_time = [years_to_maturity]
    time_countdown = years_to_maturity
    while time_countdown-0.5 > 1/365.25:
        time_countdown -= 0.5
        cash_flow += [coupon/2]
        cash_time += [time_countdown]
    def yield_calc(x):
        pv = -price
        for i in range(len(cash_flow)):
            pv += cash_flow[i] / (1+x/2/100)**(2*cash_time[i])
        return pv
    return optimize.newton(yield_calc, coupon)

def cashflow_calc(price, coupon, years_to_maturity):
    cash_flow = [100+coupon/2]
    cash_time = [years_to_maturity]
    time_countdown = years_to_maturity
    while time_countdown-0.5 > 1/365.25:
        time_countdown -= 0.5
        cash_flow += [coupon/2]
        cash_time += [time_countdown]
    return pd.DataFrame([cash_time,cash_flow],index=["TTM","Cashflow"]).transpose().set_index("TTM")

def price_calc_flat_discount(price, coupon, years_to_maturity, discount_rate):
    cash_flow = [100+coupon/2]
    cash_time = [years_to_maturity]
    time_countdown = years_to_maturity
    price = (100+coupon/2)*(1+discount_rate/2)**(-years_to_maturity)
    while time_countdown-0.5 > 1/365.25:
        time_countdown -= 0.5
        cash_flow += [coupon/2]
        cash_time += [time_countdown]
        price += coupon/2*(1+discount_rate/2)**(-time_countdown)
    return pd.DataFrame([cash_time,cash_flow],index=["TTM","Cashflow"]).transpose().set_index("TTM")

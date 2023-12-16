def logistic_ode(alpha, beta, P0, P_infty, dt, max_time):
    P = P0
    t = 0
    while t < max_time:
        if P >= P_infty:
            return t
        P += (alpha * P - beta * P**2) * dt
        t += dt
    return None

# Scenarios
scenarios = [
    (0.1, 0.001, 2, 29.75),
    (0.1, 0.001, 2, 115.35),
    (0.1, 0.0001, 2, 115.35),
    (0.01, 0.001, 2, 155.346),
    (0.1, 0.01, 100, 155.346)
]

# Parameters
dt = 0.01  # Time step
max_time = 1000  # Maximum time to simulate

# Analyzing each scenario
for i, (alpha, beta, P0, P_infty) in enumerate(scenarios, start=1):
    time_to_reach = logistic_ode(alpha, beta, P0, P_infty, dt, max_time)
    if time_to_reach is not None:
        print(f"Scenario {i}: Population reaches {P_infty} at time {time_to_reach} units.")
    else:
        print(f"Scenario {i}: Population does not reach {P_infty} within {max_time} time units.")

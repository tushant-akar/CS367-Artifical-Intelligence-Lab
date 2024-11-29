import numpy as np
from scipy.stats import poisson
from functools import lru_cache

# Constants
lambda_r1 = 3
lambda_r2 = 4
lambda_R1 = 3
lambda_R2 = 2
discount_factor = 0.9
max_bikes = 20
max_move = 5
reward_per_rental = 10
move_cost = 2

# Initialize value function and policy arrays
V = np.zeros((max_bikes + 1, max_bikes + 1))  # Value function
policy = np.zeros((max_bikes + 1, max_bikes + 1), dtype=int)  # Initial policy (0: no bikes moved)
delta = float('inf')

# Precompute Poisson probabilities for rental and return values (0 to max_bikes)
poisson_probs_r1 = [poisson.pmf(i, lambda_r1) for i in range(max_bikes + 1)]
poisson_probs_r2 = [poisson.pmf(i, lambda_r2) for i in range(max_bikes + 1)]
poisson_probs_R1 = [poisson.pmf(i, lambda_R1) for i in range(max_bikes + 1)]
poisson_probs_R2 = [poisson.pmf(i, lambda_R2) for i in range(max_bikes + 1)]

# Memoization for expected value computation
@lru_cache(None)
def compute_expected_value(b1, b2, a):
    expected_value = 0
    # Loop only through feasible rental values for r1, r2
    max_r1 = min(b1 - a, max_bikes)  # Max rental at location 1 after movement
    max_r2 = min(b2 + a, max_bikes)  # Max rental at location 2 after movement
    
    for r1 in range(max_r1 + 1):  # Rental at location 1
        for r2 in range(max_r2 + 1):  # Rental at location 2
            rented_bikes1 = min(b1 - a, r1)
            rented_bikes2 = min(b2 + a, r2)

            # Loop only through feasible return values for R1, R2
            max_R1 = rented_bikes1  # Max return at location 1
            max_R2 = rented_bikes2  # Max return at location 2
            
            for R1 in range(max_R1 + 1):  # Return at location 1
                for R2 in range(max_R2 + 1):  # Return at location 2
                    # Calculate the next state after rent and return
                    new_b1 = min(max_bikes, (b1 - a - rented_bikes1 + R1))
                    new_b2 = min(max_bikes, (b2 + a - rented_bikes2 + R2))

                    # Calculate the reward
                    rental_income = rented_bikes1 * reward_per_rental + rented_bikes2 * reward_per_rental
                    move_cost_total = abs(a) * move_cost
                    reward = rental_income - move_cost_total

                    # Probability of the transition
                    prob = poisson_probs_r1[r1] * poisson_probs_r2[r2] * poisson_probs_R1[R1] * poisson_probs_R2[R2]
                    expected_value += prob * (reward + discount_factor * V[new_b1, new_b2])
    
    return expected_value

# Iterate until convergence
iteration = 0
while delta > 1e-6:
    iteration += 1
    delta = 0
    print(f"\nIteration {iteration}: Starting value function update.")
    
    # Loop through all states (b1, b2)
    for b1 in range(max_bikes + 1):
        for b2 in range(max_bikes + 1):
            # Store the current value function
            v_old = V[b1, b2]
            best_value = float('-inf')
            best_action = 0  # Default action (no bike movement)

            print(f"  Processing state ({b1}, {b2})...")
            
            # Loop through all possible actions (-5 to 5)
            for a in range(-max_move, max_move + 1):
                if 0 <= b1 - a <= max_bikes and 0 <= b2 + a <= max_bikes:
                    # Compute expected value using the precomputed Poisson probabilities and memoization
                    expected_value = compute_expected_value(b1, b2, a)

                    # Check if this action yields a better value
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = a
                        
            # Update the value function and policy
            V[b1, b2] = best_value
            policy[b1, b2] = best_action
            delta = max(delta, abs(v_old - V[b1, b2]))

            print(f"    Best action: {best_action} with expected value: {best_value:.4f}")

    print(f"End of iteration {iteration}. Delta: {delta:.4f}")

print("Optimal value function and policy computed.")

# Print final policy
print("\nFinal Policy (number of bikes to move from Location 1 to Location 2):")
print(policy)

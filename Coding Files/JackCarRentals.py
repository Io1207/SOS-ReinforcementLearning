import numpy as np

max_cars = 20
max_transfer = 5
rental_reward = 10
transfer_cost = 2
poisson_cache = {}

#Poisson distribution 
def poisson(n, lam):
    global poisson_cache
    key = (n, lam)
    if key not in poisson_cache:
        poisson_cache[key] = np.exp(-lam) * (lam ** n) / np.math.factorial(n)
    return poisson_cache[key]

# expected return func
def expected_return(state, action, state_value):
    returns = 0.0
    
    # mooove
    cars_loc1 = min(state[0] - action, max_cars)
    cars_loc2 = min(state[1] + action, max_cars)
    returns -= transfer_cost * abs(action)
    
    for rental_requests_loc1 in range(0, 11):
        for rental_requests_loc2 in range(0, 11):
            prob_rentals = poisson(rental_requests_loc1, 3) * poisson(rental_requests_loc2, 4)
            
            valid_rentals_loc1 = min(cars_loc1, rental_requests_loc1)
            valid_rentals_loc2 = min(cars_loc2, rental_requests_loc2)
            
            reward = (valid_rentals_loc1 + valid_rentals_loc2) * rental_reward
            
            cars_loc1_left = cars_loc1 - valid_rentals_loc1
            cars_loc2_left = cars_loc2 - valid_rentals_loc2
            
            returned_cars_loc1 = np.random.poisson(3)
            returned_cars_loc2 = np.random.poisson(2)
            
            cars_loc1_end = min(cars_loc1_left + returned_cars_loc1, max_cars)
            cars_loc2_end = min(cars_loc2_left + returned_cars_loc2, max_cars)
            
            returns += prob_rentals * (reward + state_value[cars_loc1_end, cars_loc2_end])
    
    return returns

# iterate
def policy_iteration():
    state_value = np.zeros((max_cars + 1, max_cars + 1))
    policy = np.zeros((max_cars + 1, max_cars + 1), dtype=np.int)
    
    stable_policy = False
    while not stable_policy:
        # eval
        while True:
            old_value = np.copy(state_value)
            for i in range(max_cars + 1):
                for j in range(max_cars + 1):
                    state_value[i, j] = expected_return([i, j], policy[i, j], state_value)
            if np.sum(np.abs(old_value - state_value)) < 1e-4:
                break
        
        stable_policy = True
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in range(-max_transfer, max_transfer + 1):
                    if 0 <= i - action <= max_cars and 0 <= j + action <= max_cars:
                        action_returns.append(expected_return([i, j], action, state_value))
                    else:
                        action_returns.append(float('-inf'))
                new_action = np.argmax(action_returns) - max_transfer
                policy[i, j] = new_action
                if new_action != old_action:
                    stable_policy = False
    
    return policy, state_value

if __name__ == "__main__":
    policy, state_value = policy_iteration()
    print("best policy:")
    print(policy)
    print("optimal state val:")
    print(state_value)

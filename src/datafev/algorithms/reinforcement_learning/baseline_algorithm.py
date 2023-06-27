# TODO - Create an algorithm which interacts with the same simulation environment and picks
#  where to go at any given time interval based on the following:
#  - Minimize the total time spent charging
#  - Minimize the total time spent driving
#  The intent is to use this as a comparison to the RL model's choices
#  This is based on Tesla's "Trip Planner"
import math

from geolocation.maps_free import get_distance_and_time

times = []
visited_chargers = []

def baseline(environment, max_attempts):
    environment.tracking_baseline = True
    environment.reset()

    attempts = 0

    make, model, cur_soc, max_soc, base_soc, cur_lat, cur_long, org_lat, org_long, dest_lat, dest_long, *_ = environment.state
    usage_per_hour, charge_per_hour = environment.ev_info()

    populate_options(environment)
    done = False

    # Keep travelling to chargers or destination if you can get to it
    while done is not True and attempts < max_attempts:
        attempts += 1
        max_distance = environment.cur_soc / (usage_per_hour / 3600)

        done = check_done(environment, max_distance)

        if done is not True:
            current_best = find_best_charger(environment, max_distance)
            # How much to charge
            target_soc = (usage_per_hour / 3600) * current_best[2]

            # Go to the charger
            while environment.is_charging is not True and done is not True:
                next_state, reward, done = environment.step(current_best[0])  # Execute action

            # Charge until there's enough to travel to destination
            while environment.cur_soc < target_soc and done is not True:
                next_state, reward, done = environment.step(current_best[0])  # Execute action

            # Repopulate the options
            populate_options(environment)

def find_best_charger(environment, max_distance):
    # Pick the best charger to go to
    current_best = (0, math.inf, math.inf)
    for i in range(len(environment.charger_coords) - 1):
        total_time = times[i + 1][1] + times[i + 1][2]
        if (i + 1) not in visited_chargers and total_time < current_best[1] + current_best[2] and max_distance > times[i + 1][1]:
            current_best = (i + 1, times[i + 1][1], times[i + 1][2])

    visited_chargers.append(current_best[0])

    return current_best

def populate_options(environment):
    make, model, cur_soc, max_soc, base_soc, cur_lat, cur_long, org_lat, org_long, dest_lat, dest_long, *_ = environment.state

    times.clear()

    # Time for option 0
    times.append((0, get_distance_and_time((cur_lat, cur_long), (dest_lat, dest_long))[1], 0))

    for i in range(len(environment.charger_coords)):
        times.append((i + 1,
                      get_distance_and_time((cur_lat, cur_long),
                                            (environment.charger_coords[i][1], environment.charger_coords[i][2]))[1],
                      get_distance_and_time((environment.charger_coords[i][1], environment.charger_coords[i][2]),
                                            (dest_lat, dest_long))[1]))

def check_done(environment, max_distance):
    # Check if simulation can reach destination
    if times[0][1] < max_distance:
        done = False
        while done is not True:
            next_state, reward, done = environment.step(0)  # Execute action
        return True
    else:
        return False
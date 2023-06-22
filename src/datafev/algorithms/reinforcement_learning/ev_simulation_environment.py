import copy
import csv
import math

from geolocation.maps_free import get_closest_chargers, move_towards, get_distance_and_time
from geolocation.visualize import read_excel_data

# TODO - Create simulation environment which is capable of:
#  - Introducing randomness (traffic will randomly fluctuate at charging stations based on time-of-day)
#  - Considering different makes and mmodels and decreasing (or increasing while charging) an EVs SoC propotionately
#  - Considering the load placed on each charging station relative to the traffic at the station
#  - Simulating the travel from an origin to destination

class EVSimEnvironment:
    def __init__(
            self,
            num_of_episodes,
            num_of_chargers,
            make,
            model,
            cur_soc,
            max_soc,
            org_lat,
            org_long,
            dest_lat,
            dest_long
    ):
        """Create environment

        Args:
            num_of_chargers: Amount of chargers to consider
            make: Make of EV
            model: Model of EV
            cur_soc: Current state of Charge of EV in Watts
            max_soc: Maximum SoC capability in Watts
            org_lat: Latitude of route origin
            org_long: Longitude of route origin
            dest_lat: Latitude of route destination
            dest_long: Longitude of route destination

        Returns:
            Environment to use for EV simulations
        """

        self.num_of_episodes = num_of_episodes
        self.episode_num = 0
        self.visited_list = []
        self.current_path = []
        self.best_path = []

        self.step_num = 0
        self.episode_reward = 0

        self.max_reward = math.inf * -1

        self.prev_distance = 0

        self.prev_charging = False

        self.num_of_chargers = num_of_chargers
        self.make = make
        self.model = model
        self.cur_soc = cur_soc
        self.max_soc = max_soc
        self.base_soc = cur_soc # Used to reset
        self.cur_lat = org_lat
        self.cur_long = org_long
        self.org_lat = org_lat
        self.org_long = org_long
        self.dest_lat = dest_lat
        self.dest_long = dest_long

        self.usage_per_hour, self.charge_per_hour = self.ev_info()

        self.is_charging = False

        file_path = 'data/charging_stations.xlsx'
        sheet_name = 'Station Info'
        self.charger_info = read_excel_data(file_path, sheet_name)

        self.charger_lat = self.charger_info.iloc[:, 1].tolist()  # Extract values from the 3rd column
        self.charger_long = self.charger_info.iloc[:, 2].tolist()  # Extract values from the 4th column

        # Update state
        self.update_state()

    def step(self, action):
        """Simulate a step in the EVs travel

        Args:
            action: Number to use as index for stations list

        Returns:
            next_state: New state of the system
            reward: Reward for current state
            done: Indicator for if simulation is done
        """

        self.step_num += 1

        # Log every tenth episode
        if self.episode_num % math.ceil(self.num_of_episodes / 10) == 0:
            self.log(action)

        current_state = copy.copy(self.state)
        done = False
        if self.is_charging == False:
            # Consume battery while driving
            self.cur_soc -= self.usage_per_hour / (4 * 60)
            if action == 0:
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), 15)
                # Find how far destination is away from current coordinates in minutes
                time_to_destination = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[1] / 60
                # Start charging EV if within 15 minutes of charging station
                if time_to_destination <= 15 / 60:
                    done = True
                    if self.episode_num % math.ceil(self.num_of_episodes / 10) == 0:
                        self.log(action, True)
            else:
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.charger_info.loc[action - 1].latitude, self.charger_info.loc[action - 1].longitude), 15)
                # Find how far station is away from current coordinates in minutes
                time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_info.loc[action - 1].latitude, self.charger_info.loc[action - 1].longitude))[1] / 60
                # Start charging EV if within 15 minutes of charging station
                if time_to_station <= 15 / 60:
                    self.is_charging = True

            if self.cur_soc <= 0:
                done = True
        else:
            # Increase battery while charging
            self.cur_soc += self.charge_per_hour / (4 * 60)
            if self.cur_soc > self.max_soc: self.cur_soc = self.max_soc
            if action != 0:
                time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_info.loc[action - 1].latitude, self.charger_info.loc[action - 1].longitude))[1] / 60

            if action == 0 or self.cur_soc >= self.max_soc or time_to_station > 15 / 60:
                self.is_charging = False
                self.prev_charging = False
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), 15)
            else:
                self.prev_charging = True

        # Update state
        self.update_state()

        reward = self.reward(current_state, done)

        self.episode_reward += reward

        return self.state, reward, done

    def log(self, action, final = False):
        new_row = []
        new_row.append(self.episode_num)
        new_row.append(action)
        new_row.append(self.step_num)
        new_row.append(round(self.cur_soc / 1000, 2))
        new_row.append(self.is_charging)
        new_row.append(round(self.episode_reward, 2))
        if final == False:
            new_row.append(self.cur_lat)
            new_row.append(self.cur_long)
        else:
            new_row.append(self.dest_lat)
            new_row.append(self.dest_long)

        for charger in self.charger_coords:
            new_row.append(charger[1])
            new_row.append(charger[2])

        self.current_path.append(new_row)
        self.visited_list.append(new_row)

    # Reset all states
    def reset(self):
        self.step_num = 0

        if self.episode_reward > self.max_reward:
            self.best_path = self.current_path

        self.episode_reward = 0

        self.cur_soc = self.base_soc
        self.cur_lat = self.org_lat
        self.cur_long = self.org_long

        self.current_path = []

        self.episode_num += 1

        # Update state
        self.update_state()

        return self.state

    # Get reward
    def reward(self, state, done):
        #  +1 for each charging station within range
        #  +50 for being able to make it to the destination
        #  +1-10 for being closer to 80% SoC
        #  -50 for charging when above 80% battery
        #  +25 for reaching a charging station with less than 20% battery
        #  +15 for continuing to charge when below 80% battery
        #  -15 for moving away from the destination
        #  -1 for each timestep
        #  -100000 and done if run out of SoC
        #  +100000 for reaching the destination
        #  +1 for each percentage closer to destination from origin

        make, model, cur_soc, max_soc, base_soc, cur_lat, cur_long, org_lat, org_long, dest_lat, dest_long, *_ = state
        usage_per_hour, charge_per_hour = self.ev_info()

        reward = 0

        reward -= self.step_num

        for i in range(len(self.charger_coords)):
            distance_to_station, time_to_station = get_distance_and_time((cur_lat, cur_long), (self.charger_coords[i][0], self.charger_coords[i][1]))
            time_to_station /= 3600 # Convert to hours
            # Increase reward by 10 for each charging station within range
            if usage_per_hour * time_to_station < cur_soc:
                reward += 1

        if self.is_charging and self.prev_charging == False and cur_soc / max_soc < 0.2:
            reward += 25

        if self.is_charging and self.prev_charging and cur_soc / max_soc < 0.8:
            reward += 15

        if self.is_charging and cur_soc / max_soc > 0.8:
            reward -= 50

        distance_from_origin, time_from_origin = get_distance_and_time((org_lat, org_long), (dest_lat, dest_long))
        distance_to_dest, time_to_dest = get_distance_and_time((cur_lat, cur_long), (dest_lat, dest_long))
        time_to_dest /= 3600  # Convert to hours
        # Increase reward by 50 if able to get to destination
        if usage_per_hour * time_to_dest < cur_soc:
            reward += 50

        # +1 for each percentage closer to destination from origin
        if distance_to_dest == 0:
            reward += 100
        elif distance_from_origin < distance_to_dest:
            reward += -distance_to_dest / distance_from_origin
        else:
            reward += 100 * (1 - (distance_to_dest / distance_from_origin))

        if self.prev_distance != 0:
            if self.prev_distance <= distance_to_dest:
                reward -= 15

        self.prev_distance = distance_to_dest

        battery_percentage = self.cur_soc / self.max_soc
        reward += min(max(1 + 9 * abs(battery_percentage - 0.8), 1), 10)

        if self.cur_soc <= 0:
            reward -= 100000

        if time_to_dest < 15 and done:
            reward += 100000

        return reward

    # TODO - Get EV Info
    def ev_info(self):
        # TODO - Make estimates more realistic using LH Dataset
        usage_per_hour = 15600 * 60 # Average usage per hour of Tesla
        charge_per_hour = 12500 * 60 # Average charge per hour of Tesla
        return usage_per_hour, charge_per_hour

    def update_state(self):

        list_of_chargers = list(zip(self.charger_lat, self.charger_long))
        list_of_chargers = [(i, val1, val2) for i, (val1, val2) in enumerate(list_of_chargers)]
        self.charger_coords = get_closest_chargers(self.cur_lat, self.cur_long, self.num_of_chargers, list_of_chargers)

        self.state = (self.make, self.model, self.cur_soc, self.max_soc, self.base_soc, self.cur_lat, self.cur_long,
                      self.org_lat, self.org_long, self.dest_lat, self.dest_long, self.charger_coords)

        unwrapped_values = sum(self.state[-1], ())
        self.state = self.state[:-1] + unwrapped_values

    def get_state_action_dimension(self):
        states = len(self.state)
        actions = 1 + self.num_of_chargers
        return states, actions

    def write_path_to_csv(self, path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            header_row = []
            header_row.append('Episode Num')
            header_row.append('Action')
            header_row.append('Timestep')
            header_row.append('SoC')
            header_row.append('Is Charging')
            header_row.append('Episode Reward')
            header_row.append('Latitude')
            header_row.append('Longitude')

            for i in range(self.num_of_chargers):
                header_row.append(f"Charger {i + 1} Latitude")
                header_row.append(f"Charger {i + 1} Longitude")

            writer.writerow(header_row)

            for row in self.visited_list:
                writer.writerow(row)

            for row in self.best_path:
                row[0] = "Best"
                writer.writerow(row)

    def print(self):
        print(self.state)
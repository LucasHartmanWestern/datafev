import copy
import csv
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

        self.episode_num = 0
        self.visited_list = []

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

        self.visited_list.append(self.episode_num, self.cur_lat, self.cur_long)

        current_state = copy.copy(self.state)
        done = False
        if self.is_charging == False:
            # Consume battery while driving
            self.cur_soc -= self.usage_per_hour
            if action == 0:
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), 15)
                # Find how far destination is away from current coordinates in minutes
                time_to_destination = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_info.loc[action].latitude, self.charger_info.loc[action].longitude))[1] / 60
                # Start charging EV if within 15 minutes of charging station
                if time_to_destination <= 15:
                    done = True
            else:
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.charger_info.loc[action].latitude, self.charger_info.loc[action].longitude), 15)
                # Find how far station is away from current coordinates in minutes
                time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_info.loc[action].latitude, self.charger_info.loc[action].longitude))[1] / 60
                # Start charging EV if within 15 minutes of charging station
                if time_to_station <= 15:
                    self.is_charging = True
        else:
            # Increase battery while charging
            self.cur_soc += self.charge_per_hour
            if action == 0:
                self.is_charging = False
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), 15)

        # Update state
        self.update_state()

        return self.state, self.reward(current_state), done

    # Reset all states
    def reset(self):
        self.cur_soc = self.base_soc
        self.cur_lat = self.org_lat
        self.cur_long = self.org_long

        self.episode_num += 1

        # Update state
        self.update_state()

        return self.state

    # Get reward
    def reward(self, state):
        #  +10 for each charging station within range
        #  +100 for being able to make it to the destination
        #  +1-10 for being closer to 80% SoC

        make, model, cur_soc, max_soc, base_soc, cur_lat, cur_long, org_lat, org_long, dest_lat, dest_long, *_ = state
        usage_per_hour, charge_per_hour = self.ev_info()

        reward = 0

        for i in range(len(self.charger_coords)):
            distance_to_station, time_to_station = get_distance_and_time((cur_lat, cur_long), (self.charger_coords[i][0], self.charger_coords[i][1]))
            time_to_station /= 3600 # Convert to hours
            # Increase reward by 10 for each charging station within range
            if usage_per_hour * time_to_station < cur_soc:
                reward += 10

        distance_to_dest, time_to_dest = get_distance_and_time((cur_lat, cur_long), (dest_lat, dest_long))
        time_to_dest /= 3600  # Convert to hours
        # Increase reward by 100 if able to get to destination
        if usage_per_hour * time_to_dest < cur_soc:
            reward += 100

        battery_percentage = self.cur_soc / self.max_soc
        reward += min(max(1 + 9 * abs(battery_percentage - 0.8), 1), 10)

        return reward

    # TODO - Get EV Info
    def ev_info(self):
        # TODO - Make estimates more realistic using LH Dataset
        usage_per_hour = 156000 # Average usage per hour of Tesla
        charge_per_hour = 125000 # Average charge per hour of Tesla
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

    def write_path_to_csv(self):
        with open('outputs/routes.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode Num', 'Latitude', 'Longitude'])

            for row in self.visited_list:
                writer.writerow(row)

    def print(self):
        print(self.state)
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
            make,
            model,
            soc,
            cur_lat,
            cur_long,
            org_lat,
            org_long,
            dest_lat,
            dest_long
    ):
        """Create environment

        Args:
            make: Make of EV
            model: Model of EV
            soc: State of Charge of EV
            cur_lat: Latitude of EV currently
            cur_long: Longitude of EV currently
            org_lat: Latitude of route origin
            org_long: Longitude of route origin
            dest_lat: Latitude of route destination
            dest_long: Longitude of route destination

        Returns:
            Environment to use for EV simulations
        """

        file_path = 'data/charging_stations.xlsx'
        sheet_name = 'Station Info'
        charger_info = read_excel_data(file_path, sheet_name)

    # TODO - Simulate an EVs travel
    def step(self, action):
        return

    # TODO - Reset all states
    def reset(self):
        return

    # TODO - Get reward
    def reward(self):
        return

    # TODO - Get EV Info
    def ev_info(self):
        return
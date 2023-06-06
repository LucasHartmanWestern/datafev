import math

from maps import get_distance_and_time, move_towards

origin = "400 Lyle St, London, ON"
destination = "7720 Patrick St, Port Franks, ON"

dist1 = get_distance_and_time(origin, destination)

print(dist1)

update_loc = move_towards(origin, destination, 10)
print(update_loc)
dist2 = get_distance_and_time(update_loc, destination)
print(dist2)

while(dist2[0] > 10):
    update_loc = move_towards(update_loc, destination, 25)
    print(update_loc)
    dist2 = get_distance_and_time(update_loc, destination)
    print(dist2)
import math

# Define a class to represent a 2D point
class Point(object):
    def __init__(self, x, y):
        # Initialize x and y coordinates for the point
        self.x = x
        self.y = y

    # Provide a string representation for the Point
    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)

    # Define equality based on x and y values
    def __eq__(self, other):
        return other.x == self.x and other.y == self.y

    # Define hash function to allow Point objects to be used in sets and dictionaries
    def __hash__(self):
        return hash((self.x, self.y))

# Define a class to represent the state in A* algorithm
class State(object):
    def __init__(self, point, g, h, parent=None):
        self.point = point       # Current point in the state
        self.g = g               # Cost from the start node to this point
        self.h = h               # Heuristic estimate from this point to the goal
        self.f = self.g + self.h # Total estimated cost
        self.parent = parent     # Parent point/state

    # Compare states based on f value for priority
    def __lt__(self, other):
        return self.f < other.f

    # Provide a string representation for the State
    def __repr__(self):
        return '({}, g={}, h={})'.format(self.point, self.g, self.h)

# Function to calculate Euclidean distance between two points
def distance(a, b):
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

# Function to check if two line segments intersect
def intersect(a, b, c, d):
    return (
        # Bounding box overlap check for the segments
        min(a.x, b.x) < max(c.x, d.x) and
        min(c.y, d.y) < max(a.y, b.y) and
        min(c.x, d.x) < max(a.x, b.x) and
        min(a.y, b.y) < max(c.y, d.y) and
        # Intersection test using cross product
        (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y) != 0 and
        (d.y - a.y) * (b.x - a.x) - (d.x - a.x) * (b.y - a.y) != 0
    )

# Function to parse input file and extract relevant data
def parse_file(fname):
    obstacles = [] # List of obstacle polygons
    vertices = set() # Unique vertices from the obstacles

    # Open and read the file
    with open(fname, 'r') as f:
        lines = f.readlines()
        # Parse start and goal points
        start_x, start_y = map(int, lines[0].split())
        goal_x, goal_y = map(int, lines[1].split())
        start_point = Point(start_x, start_y)
        goal_point = Point(goal_x, goal_y)

        # Parse obstacle vertices and add to list
        for line in lines[3:]:
            obstacle_points = [Point(x, y) for x, y in zip(*[iter(map(int, line.split()))]*2)]
            obstacles.append(obstacle_points)
            for point in obstacle_points:
                vertices.add(point)

    return start_point, goal_point, obstacles, vertices

# Check if a straight path between two points is blocked by any obstacle
def can_connect(a, b, obstacles):
    for obstacle in obstacles:
        for i in range(len(obstacle)):
            if intersect(a, b, obstacle[i], obstacle[(i + 1) % len(obstacle)]):
                return False
    return True

# Main A* search function
def search(start_point, goal_point, vertices, obstacles):
    open_dict = {start_point: State(start_point, 0, distance(start_point, goal_point))}
    closed_dict = {}

    # While there are nodes to explore
    while open_dict:
        # Get the node with the lowest f value
        current_state = min(open_dict.values())
        del open_dict[current_state.point]

        # If the current node is the goal, we are done
        if current_state.point == goal_point:
            closed_dict[current_state.point] = current_state
            break

        # Explore neighbors
        for next_point in vertices:
            # Skip the current point and points blocked by obstacles
            if next_point == current_state.point or not can_connect(current_state.point, next_point, obstacles):
                continue

            g = current_state.g + distance(current_state.point, next_point)
            h = distance(next_point, goal_point)
            new_state = State(next_point, g, h, current_state.point)

            # Update open and closed lists with new state
            if next_point in closed_dict and closed_dict[next_point].g > g:
                closed_dict[next_point] = new_state
                open_dict[next_point] = new_state
            elif next_point not in open_dict:
                open_dict[next_point] = new_state

        closed_dict[current_state.point] = current_state

    return closed_dict

# Print the resulting path
def print_solution(closed_dict, goal_point):
    path_states = []
    curr_state = closed_dict.get(goal_point)

    # Backtrack from the goal to the start
    while curr_state:
        path_states.append(curr_state)
        curr_state = closed_dict.get(curr_state.parent)

    path_states.reverse()
    print('Point\t Cumulative Cost')
    for state in path_states:
        print('{}\t {}'.format(state.point, state.g))

# Top-level function to drive the A* search
def AStarSearch(fname):
    start_point, goal_point, obstacles, vertices = parse_file(fname)
    closed_dict = search(start_point, goal_point, vertices, obstacles)
    print_solution(closed_dict, goal_point)

# Driver code
if __name__ == '__main__':
    AStarSearch('try.txt')

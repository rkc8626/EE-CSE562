def is_valid(state):
    """
    Check if the given state is valid.
    m is the number of missionaries on the left bank.
    c is the number of cannibals on the left bank.
    b is boats on the left or right bank.
    """
    m, c, b = state

    # Check if the number of missionaries or cannibals is out of bounds.
    if not(0 <= m <= 3 and 0 <= c <= 3):
        return False

    # Check if missionaries are outnumbered by cannibals on either bank.
    if (m < c and m > 0) or (3 - m < 3 - c and 3 - m > 0):
        return False

    return True

def possible_transitions(m, c, b):
    """
    Generate possible transitions based on current state.
    """
    # Determine the maximum number of each that can be moved.
    max_m = m if b == 'L' else 3 - m
    max_c = c if b == 'L' else 3 - c

    # Generate possible transitions.
    transitions = [(dm, dc) for dm in range(max_m + 1) for dc in range(max_c + 1) if 1 <= dm + dc <= 2]

    return transitions

def dfs(state, path, solutions, counts):
    """
    Depth-first search to find solutions.
    """
    m, c, b = state

    # Goal state reached.
    if state == (0, 0, 'R'):
        solutions.append(path.copy())
        counts['total'] += 1
        return

    # Check if the current state is a repeated state.
    if state in path:
        counts['repeated'] += 1
        return

    path.append(state)
    counts['total'] += 1

    # Generate next possible states based on transitions.
    for dm, dc in possible_transitions(m, c, b):
        new_state = (m - dm, c - dc, 'R') if b == 'L' else (m + dm, c + dc, 'L')

        if is_valid(new_state):
            dfs(new_state, path, solutions, counts)
        else:
            counts['illegal'] += 1

    path.pop()

def solve():
    """
    Solve the Missionaries and Cannibals problem.
    """
    initial_state = (3, 3, 'L')
    solutions = []
    counts = {'illegal': 0, 'repeated': 0, 'total': 0}
    dfs(initial_state, [], solutions, counts)

    for i, solution in enumerate(solutions):
        print(f"\nSolution {i + 1}:")
        for step in solution:
            print(step)

    print(f"Illegal states: {counts['illegal']}")
    print(f"Repeated states: {counts['repeated']}")
    print(f"Total states searched: {counts['total']}")

solve()

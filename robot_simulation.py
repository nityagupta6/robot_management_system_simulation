import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
import random
from collections import Counter, defaultdict

# Global variables
is_paused = True
has_conflict = False
step_index = 0
move_history = []
num_robots = 3  # Default number of robots


def create_grid_graph(grid_size=10):
    G = nx.grid_2d_graph(grid_size, grid_size)
    
    pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Standard layout

    mapping = {
        node: idx for idx, node in enumerate(
            sorted(G.nodes(), key=lambda n: (n[1], -n[0]))
        )
    }

    G = nx.relabel_nodes(G, mapping)
    pos = {mapping[node]: coord for node, coord in pos.items()}

    return G, pos

# Assign robots to random nodes and assign random goals
def assign_robots(G, num_robots):
    nodes = list(G.nodes())
    robots = {}
    for i in range(num_robots):
        start = random.choice(nodes)
        goal = random.choice(nodes)
        while goal == start:
            goal = random.choice(nodes)
        robots[i] = {"position": start, "goal": goal, "name": f"R{i+1}", "stuck": False}
    return robots

def move_robots(G, robots):
    global has_conflict
    moves = []
    
    # Track which robots are involved in conflicts
    conflicts = detect_conflicts(robots)
    has_conflict = bool(conflicts)
    
    # For each robot, track their movement and "stuck" state
    for robot_id, robot in robots.items():
        if robot["position"] in conflicts:
            robot["stuck"] = True  # Mark the robot as stuck if it's in conflict
        elif not robot["stuck"]:
            current_node = robot["position"]
            goal_node = robot["goal"]
            if current_node != goal_node:
                path = nx.shortest_path(G, source=current_node, target=goal_node)
                next_position = path[1]
                moves.append((robot_id, current_node, next_position, robot["stuck"]))  # Track the robot ID, position, next position, and stuck state
                robot["position"] = next_position
            else:
                new_goal = random.choice(list(G.nodes()))
                while new_goal == current_node:
                    new_goal = random.choice(list(G.nodes()))
                robot["goal"] = new_goal
                moves.append((robot_id, current_node, current_node, robot["stuck"]))  # Track robot staying in place and stuck state
    
    return moves



def reverse_moves(robots, moves):
    # Restore positions and stuck state from the history
    for robot_id, old_pos, new_pos, stuck_state in moves:
        robots[robot_id]["position"] = old_pos
        robots[robot_id]["stuck"] = stuck_state  # Restore the "stuck" state

def detect_conflicts(robots):
    positions = [robot["position"] for robot in robots.values()]
    count = Counter(positions)
    conflicting_nodes = {node for node, freq in count.items() if freq > 1}
    return conflicting_nodes


def update_plot(ax, scatter_nodes, scatter_robots, robot_labels, node_colors, G, pos, robots, conflicts, header_text):
    for node in G.nodes():
        node_colors[node] = "lightblue"

    for conflict_node in conflicts:
        node_colors[conflict_node] = "red"

    occupied_nodes = defaultdict(list)
    for robot_id, robot in robots.items():
        current_node = robot["position"]
        if current_node not in conflicts:
            node_colors[current_node] = "green"
        occupied_nodes[current_node].append(robot_id)

    for label in robot_labels.values():
        label.set_text("")

    for node, robot_ids in occupied_nodes.items():
        robot_numbers = ",".join([f"R{robot_id+1}" for robot_id in robot_ids])
        label_pos = (pos[node][0], pos[node][1] - 0.3)
        robot_labels[node].set_position(label_pos)
        robot_labels[node].set_text(robot_numbers)

    for artist in ax.texts[:]:
        if artist.get_bbox_patch() and artist.get_bbox_patch().get_facecolor() == (1.0, 1.0, 0.0, 1.0):
            artist.remove()

    goal_nodes = defaultdict(list)
    for robot_id, robot in robots.items():
        goal_nodes[robot["goal"]].append(f"G{robot_id+1}")

    for goal_node, goal_labels in goal_nodes.items():
        combined_goal_label = ",".join(goal_labels)
        goal_pos = (pos[goal_node][0], pos[goal_node][1] + 0.3)
        ax.text(
            goal_pos[0], goal_pos[1], combined_goal_label,
            fontsize=10,
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.3')
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=list(node_colors.values()), node_size=500, edgecolors="gray"
    )

    header_info = "\n".join(
        [f"{robot['name']}: Current {robot['position']}, Goal G{robot_id+1} ({robot['goal']})" for robot_id, robot in robots.items()]
    )
    header_text.set_text(header_info)

    plt.draw()


def reset_simulation_factory(G, pos, ax, robot_labels, node_colors, header_text, text_box):
    def reset_simulation(event):
        global robots, step_index, move_history, is_paused, has_conflict, num_robots

        try:
            num_robots = int(text_box.text)
        except ValueError:
            print("Invalid input for the number of robots. Resetting to default.")
            num_robots = 3

        is_paused = True
        has_conflict = False
        step_index = 0
        move_history = []

        robots = assign_robots(G, num_robots)

        conflicts = detect_conflicts(robots)
        update_plot(
            ax, None, None, robot_labels, node_colors, G, pos, robots, conflicts, header_text
        )
    return reset_simulation


def simulate_live(grid_size=10, num_robots_arg=5, steps=100):
    global is_paused, step_index, move_history, robots, G, pos, node_colors, robot_labels, num_robots

    num_robots = num_robots_arg

    G, pos = create_grid_graph(grid_size)
    robots = assign_robots(G, num_robots)
    node_colors = {node: "lightblue" for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        edge_color="gray",
    )

    robot_labels = {
        node: ax.text(
            pos[node][0] + 0.3,
            pos[node][1] - 0.8,
            "",
            fontsize=10,
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
        )
        for node in G.nodes()
    }

    header_ax = plt.axes([0.01, 0.75, 0.2, 0.2])
    header_ax.axis("off")
    header_text = header_ax.text(
        0, 1, "", fontsize=10, ha="left", va="top", transform=header_ax.transAxes
    )

    play_ax = plt.axes([0.6, 0.05, 0.1, 0.075])
    pause_ax = plt.axes([0.71, 0.05, 0.1, 0.075])
    reset_ax = plt.axes([0.82, 0.05, 0.1, 0.075])
    text_box_ax = plt.axes([0.75, 0.9, 0.2, 0.05])  # Top-right corner for the text box
    play_button = Button(play_ax, "Play")
    pause_button = Button(pause_ax, "Pause")
    reset_button = Button(reset_ax, "Reset")
    text_box = TextBox(text_box_ax, "Num Robots", initial=str(num_robots))

    def play(event):
        global is_paused
        is_paused = False

    def pause(event):
        global is_paused
        is_paused = True

    play_button.on_clicked(play)
    pause_button.on_clicked(pause)

    reset_button.on_clicked(reset_simulation_factory(G, pos, ax, robot_labels, node_colors, header_text, text_box))

    def on_key(event):
        global step_index, move_history, is_paused

        if is_paused:
            if event.key == "right":
                if step_index < steps:
                    moves = move_robots(G, robots)
                    move_history.append(moves)
                    step_index += 1
                    conflicts = detect_conflicts(robots)
                    update_plot(
                        ax, None, None, robot_labels, node_colors, G, pos, robots, conflicts, header_text
                    )
            elif event.key == "left":
                if step_index > 0:
                    moves = move_history.pop()
                    reverse_moves(robots, moves)
                    step_index -= 1
                    conflicts = detect_conflicts(robots)
                    update_plot(
                        ax, None, None, robot_labels, node_colors, G, pos, robots, conflicts, header_text
                    )

    fig.canvas.mpl_connect("key_press_event", on_key)

    def animation_step(frame):
        global step_index
        if not is_paused:
            if step_index < steps:
                moves = move_robots(G, robots)
                move_history.append(moves)
                conflicts = detect_conflicts(robots)
                update_plot(
                    ax, None, None, robot_labels, node_colors, G, pos, robots, conflicts, header_text
                )
                step_index += 1

    ani = FuncAnimation(fig, animation_step, frames=steps, interval=1500, repeat=False)
    plt.show()

if __name__ == "__main__":
    simulate_live(grid_size=10, num_robots_arg=3, steps=200)

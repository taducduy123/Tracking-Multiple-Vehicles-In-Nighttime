import numpy as np
import math
from scipy.optimize import linear_sum_assignment




# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Structure of a Headlight -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class Headlight:
    def __init__(self, x, y, a, e):
        self.x = x          # x-position
        self.y = y          # y-position
        self.a = a          # area
        self.e = e          # shape



# ----------------------------------------------------------------------------------------------------------------------
# STEP 1: Compute {s_ij} -----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Formula (9):
def compute_predicted_position(start_x, start_y, delta_x, delta_y, num_frames_taken=1):
    initial_position = np.array([start_x, start_y])  # Starting at origin (x, y)
    velocity = np.array([delta_x, delta_y])  # Velocity in x and y (units per second)
    predicted_position = initial_position + (velocity * num_frames_taken)
    return predicted_position[0], predicted_position[1]


# Formula (9):
def compute_distance_similarity(headlight_i, headlight_j):
    # Set up coefficients
    sigma = 15
    c2 = sigma * math.sqrt(2 * math.pi)

    # Compute predicted position
    P_x_i, P_y_i = compute_predicted_position(headlight_i.x, headlight_i.y, delta_x=5, delta_y=5)
    x_j, y_j = headlight_j.x, headlight_j.y

    # Compute similarity in distance
    s_d_ij = c2 * math.exp(-((P_x_i - x_j) ** 2 + (P_y_i - y_j) ** 2) / (2 * (sigma ** 2)))
    return s_d_ij


# Formula (10):
def compute_area_similarity(headlight_i, headlight_j):
    a_i, a_j = headlight_i.a, headlight_j.a

    s_a_ij = a_i / a_j
    if s_a_ij > (a_j / a_i):
        s_a_ij = a_j / a_i
    return s_a_ij


# Formula (10):
def compute_shape_similarity(headlight_i, headlight_j):
    e_i, e_j = headlight_i.e, headlight_j.e

    s_e_ij = e_i / e_j
    if s_e_ij > (e_j / e_i):
        s_e_ij = e_j / e_i
    return s_e_ij


# Formula (8):
# Compute {s_ij}
def compute_s_ij(headlights_t_minus_1, headlights_t, alpha=0.4, beta=0.3):
    # Number of headlights
    m = len(headlights_t_minus_1)
    n = len(headlights_t)


    # Initialize affinity matrix between {headlights_t_minus_1} and {headlights_t}
    s = np.zeros((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            s[i, j] = (alpha * compute_distance_similarity(headlights_t_minus_1[i], headlights_t[j])
                       + beta * compute_area_similarity(headlights_t_minus_1[i], headlights_t[j])
                       + (1 - alpha - beta) * compute_shape_similarity(headlights_t_minus_1[i], headlights_t[j]))
    return s



# ----------------------------------------------------------------------------------------------------------------------
# STEP 7: Solve the assignment problem (Tracking headlights) -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def solve_assignment_problem(matrix):
    row_ind, col_ind = linear_sum_assignment(-matrix)  # Using negative because it's maximization
    assignment_matrix = np.zeros_like(matrix)
    assignment_matrix[row_ind, col_ind] = 1
    max_energy = matrix[row_ind, col_ind].sum()
    return assignment_matrix, max_energy



# ----------------------------------------------------------------------------------------------------------------------
# STEP 12: Compute {s_vel_ij} ------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Formula (13):
def mag(l_i, l_j, l_p, l_q):
    d_ij = math.sqrt((l_i.x - l_j.x) ** 2 + (l_i.y - l_j.y) ** 2)
    d_pq = math.sqrt((l_p.x - l_q.x) ** 2 + (l_p.y - l_q.y) ** 2)

    if (d_ij ** 2) + (d_pq ** 2) > 0:
        return 2 * (d_ij * d_pq) / (d_ij ** 2 + d_pq ** 2)
    else:
        # return 2 * (d_ij * d_pq)
        return 1.0

# Formula (14):
def arctan(delta_y, delta_x):
    if delta_x == 0 and delta_y != 0:
        return math.pi / 2
    elif delta_x == 0 and delta_y == 0:
        return math.atan(delta_y)

    return math.atan(delta_y / delta_x)


# Formula (13):
def ori(l_i, l_j, l_p, l_q):

    delta_x_ij = l_j.x - l_i.x
    delta_y_ij = l_j.y - l_i.y
    delta_x_pq = l_q.x - l_p.x
    delta_y_pq = l_q.y - l_p.y

    # if delta_x_ij == 0 and delta_y_ij == 0 and delta_x_pq == 0 and delta_y_pq == 0:
    #     return 1.0

    theta_ij = arctan(delta_y_ij, delta_x_ij)
    theta_pq = arctan(delta_y_pq, delta_x_pq)

    return abs(math.cos(theta_ij - theta_pq))


# Formula (12):
def r(current_l_i, previous_l_j, current_l_gi, previous_l_gj, gamma):
    return (gamma * mag(current_l_i, previous_l_j, current_l_gi, previous_l_gj)
            + (1 - gamma) * ori(current_l_i, previous_l_j, current_l_gi, previous_l_gj))


def get_backward_headlight_index(current_headlight_index, association_matrix):
    index = -1
    for i in range(association_matrix.shape[0]):
        if association_matrix[i][current_headlight_index] == 1:
            index = i
            break
    return index


def get_forward_headlight_index(current_headlight_index, association_matrix):
    index = -1
    for i in range(association_matrix.shape[1]):
        if association_matrix[current_headlight_index][i] == 1:
            index = i
            break
    return index


# Formula (19):
# Please ensure that headlights_in_frames is the array of AT MOST 5 frames
def compute_velocity_similarity(headlights_in_frames, association_matrices, candidate_pairs, gamma):
    num_frame = len(headlights_in_frames)

    if num_frame > 5:
        print("Passing first argument for function compute_velocity_similarity is INVALID!")
        exit(0)

    n = len(candidate_pairs)
    velocity_matrix = np.zeros(n)

    index = 0
    for pair in candidate_pairs:
        #print(f"Now, we are processing pair {pair[0], pair[1]}:")
        for k in range (num_frame - 1, 0, -1):
            if k == num_frame - 1:
                current_headlight_i_index = pair[0]
                current_headlight_i = headlights_in_frames[k][current_headlight_i_index]
                #print(f"current_headlight_i = {current_headlight_i.x, current_headlight_i.y} with headlights_in_frames[k][pair[0]] = headlights_in_frames[{k}][{pair[0]}]")
                current_headlight_j_index = pair[1]
                current_headlight_j = headlights_in_frames[k][current_headlight_j_index]
                #print(f"current_headlight_j = {current_headlight_j.x, current_headlight_j.y} with headlights_in_frames[k][pair[1]] = headlights_in_frames[{k}][{pair[1]}]")

            previous_headlight_i_index = get_backward_headlight_index(current_headlight_i_index, association_matrices[k - 1])
            previous_headlight_i = headlights_in_frames[k - 1][previous_headlight_i_index]
            #print(f"previous_headlight_i = {previous_headlight_i.x, previous_headlight_i.y} with headlights_in_frames[k - 1][get_backward_headlight_index(pair[0], association_matrices[k - 1])] = headlights_in_frames[{k - 1}][{get_backward_headlight_index(pair[0], association_matrices[k - 1])}]")
            #print(f"{previous_headlight_i.x, previous_headlight_i.y} is previous headlight of {current_headlight_i.x, current_headlight_i.y}")

            previous_headlight_j_index = get_backward_headlight_index(current_headlight_j_index,association_matrices[k - 1])
            previous_headlight_j = headlights_in_frames[k - 1][previous_headlight_j_index]
            #print(f"previous_headlight_j = {previous_headlight_j.x, previous_headlight_j.y} with headlights_in_frames[k - 1][get_backward_headlight_index(pair[1], association_matrices[k - 1])] = headlights_in_frames[{k - 1}][{get_backward_headlight_index(pair[1], association_matrices[k - 1])}]")
            #print(f"{previous_headlight_j.x, previous_headlight_j.y} is previous headlight of {current_headlight_j.x, current_headlight_j.y}")

            velocity_matrix[index] += r(current_headlight_i, previous_headlight_i, current_headlight_j, previous_headlight_j, gamma)
            #print(f"(current_headlight_i, previous_headlight_i, current_headlight_j, previous_headlight_j) = ({current_headlight_i.x, current_headlight_i.y}, {previous_headlight_i.x, previous_headlight_i.y}, {current_headlight_j.x, current_headlight_j.y}, {previous_headlight_j.x, previous_headlight_j.y})")

            # RESET
            current_headlight_i = previous_headlight_i
            current_headlight_i_index = previous_headlight_i_index
            #print(f"current_headlight_i = {current_headlight_i.x, current_headlight_i.y}")
            current_headlight_j = previous_headlight_j
            #print(f"current_headlight_j = {current_headlight_j.x, current_headlight_j.y}")
            current_headlight_j_index = previous_headlight_j_index


        velocity_matrix[index] = velocity_matrix[index] / (num_frame - 1)
        index = index + 1
    return velocity_matrix



# ----------------------------------------------------------------------------------------------------------------------
# STEP 13: Update weights {w_ij} ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Formula (5):
# LEARNING GEOMETRIC MODEL
def f(delta_x, y,  delta_y, k=-0.18257495551609929, u=-47.90696625640577, s=6.153788498283703):

    # # ---------------------------- GAUSSIAN MODEL -----------------------------------
    # # Normalization Constant
    # c = 1 / (s * math.sqrt(2 * math.pi))
    # return c * math.exp((-0.5 / (s ** 2)) * ((delta_x + (k * y) - u) ** 2))

    # # ------------------------------- MY MODEL (version 1):  --------------------------------------
    # # Explain: I observe that in a particular scenario if the horizontal distance and
    # # the vertical distance between two objects below somewhat threshold, then these
    # # two objects should form a pair
    # if delta_x == 0 and delta_y == 0:
    #     return 0
    # if delta_x < 160 and delta_y < 10:          # NOTICE: This threshold may change in different scenarios
    #     return 1
    # return 0

    # # ------------------------------- MY MODEL (version 2):  --------------------------------------
    # estimation = -(k * y) + u
    # error = abs(delta_x - estimation)
    # if error < 500:
    #     print(error)
    #     return 1
    # return 0

    # ------------------------------- MY MODEL (version 3):  --------------------------------------
    if delta_x < 160 and delta_y < 10:
        c = 1 / (s * math.sqrt(2 * math.pi))
        return c * math.exp((-0.5 / (s ** 2)) * ((delta_x + (k * y) - u) ** 2))
    return 0


def get_likelihood_matrix(current_headlights, candidate_pairs):
    n = len(candidate_pairs)
    likelihoods = np.zeros(n)

    index = 0
    for pair in candidate_pairs:
        delta_x_ij = abs(current_headlights[pair[0]].x - current_headlights[pair[1]].x)
        y_ij = 0.5 * (current_headlights[pair[0]].y + current_headlights[pair[1]].y)
        delta_y_ij = abs(current_headlights[pair[0]].y - current_headlights[pair[1]].y)

        likelihoods[index] = f(delta_x_ij, y_ij, delta_y_ij)
        index = index + 1

    return likelihoods


def get_area_similarity_matrix(current_headlights, candidate_pairs):
    n = len(candidate_pairs)
    area_similarities = np.zeros(n)

    index = 0
    for pair in candidate_pairs:
        s_a_ij = compute_area_similarity(current_headlights[pair[0]], current_headlights[pair[1]])
        # print(f"{current_headlights[pair[0]].x, current_headlights[pair[0]].y} vs {current_headlights[pair[1]].x, current_headlights[pair[1]].y} = {s_a_ij}")
        area_similarities[index] = s_a_ij
        index = index + 1

    return area_similarities


# Formula (17), (18):
def update_weights(geometry_model, area_similarity, velocity_similarity, lambda_param=0.4):
    return geometry_model * (lambda_param * area_similarity + (1 - lambda_param) * velocity_similarity)



# ----------------------------------------------------------------------------------------------------------------------
# STEP 14: GRASP-based MWIS solver -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def grasp_mwis(weights, conflict_sets, candidate_pairs):
    selected = []
    weight_sum = 0
    used_nodes = set()
    result_pairs = []

    for w in sorted(weights, key=lambda x: -x[1]):  # Sort by weight descending
        i = w[0]
        if i not in used_nodes:
            # Check conflicts
            is_conflict = False
            for conflict_node in conflict_sets[i]:
                if conflict_node in used_nodes:
                    is_conflict = True
                    break
            if not is_conflict:
                selected.append(i)
                used_nodes.add(i)
                weight_sum += w[1]

    for node_id in selected:
        result_pairs.append(candidate_pairs[node_id])

    return result_pairs, weight_sum


# Create conflict sets for MWIS (Headlights in same pair can't be used twice. Ex: We don't want (1,1) be a pair)
def create_conflict_sets(num_pairs, pairs):
    conflict_sets = {i: [] for i in range(num_pairs)}

    for i in range(num_pairs):
        for j in range(i + 1, num_pairs):
            # If two pairs share the same headlight, they are in conflict
            if pairs[i][0] == pairs[j][0] or pairs[i][1] == pairs[j][1] or pairs[i][0] == pairs[j][1] or pairs[i][1] == pairs[j][0]:
                conflict_sets[i].append(j)
                conflict_sets[j].append(i)

    return conflict_sets



# ----------------------------------------------------------------------------------------------------------------------
# STEP 19: Computer r_ij based on indicators x -------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_partner_index(current_index, paring_indicator):
    partner_index = -1
    for pair in paring_indicator:
        if pair[0] == current_index:
            partner_index = pair[1]
            break
        elif pair[1] == current_index:
            partner_index = pair[0]
            break
    return partner_index


def compute_relation_similarity(headlights_t_minus_1, headlights_t, paring_indicators_t, paring_indicators_t_minus_1, gamma):
    m = len(headlights_t_minus_1)
    n = len(headlights_t)

    relation_matrix = np.zeros((m, n))

    if paring_indicators_t_minus_1 is not None:
        for i in range (m):
            for j in range (n):
                current_headlight_i = headlights_t[i]
                previous_headlight_j = headlights_t_minus_1[j]

                current_partner_i = headlights_t[get_partner_index(i, paring_indicators_t)]
                previous_partner_j = headlights_t_minus_1[get_partner_index(j, paring_indicators_t_minus_1)]

                relation_matrix[i, j] = r(current_headlight_i, previous_headlight_j, current_partner_i, previous_partner_j, gamma)

    return relation_matrix



# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Main Algorithm ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def headlight_tracking_and_pairing(headlights_in_frames, association_matrices, paring_indicators_t_minus_1, max_iterations=5):
    num_frame = len(headlights_in_frames)
    headlights_t_minus_1 = headlights_in_frames[num_frame - 2]
    headlights_t = headlights_in_frames[num_frame - 1]

    num_headlights_t_minus_1 = len(headlights_t_minus_1)
    num_headlights_t = len(headlights_t)

    # Step 1: Compute initial similarity
    similarity_matrix = compute_s_ij(headlights_t_minus_1, headlights_t)

    # Step 2: Initialize assignment matrix and energy variables
    Emax = 0
    association_matrix = np.zeros_like(similarity_matrix)
    relation_matrix = np.zeros_like(similarity_matrix)

    # Step 3: Initialize pairing variables
    svel_ij = 0
    EGmax = 0
    pairing_indicator = []

    # Create candidate pairs based on proximity (for demonstration, all-to-all pairing)
    candidate_pairs = []
    for i in range (num_headlights_t):
        for j in range (i, num_headlights_t):
            candidate_pairs.append((i, j))

    # Step 4: Start iterations
    for iter in range(1, max_iterations + 1):
        # Step 6: Update energy function E(Pi)
        energy_function = similarity_matrix + relation_matrix

        # Step 7: Solve the assignment problem using the Hungarian algorithm
        association_matrix, current_energy = solve_assignment_problem(energy_function)

        # Step 8-11:
        if current_energy > Emax:
            Emax = current_energy

        # Set gamma
        gamma = 0.5
        if iter == 1 or iter == 3 or iter == 5:
            gamma = 0.7

        # Step 12: Compute velocity similarity
        all_association_matrices = []
        for i in range(len(association_matrices)):
            all_association_matrices.append(association_matrices[i])
        all_association_matrices.append(association_matrix)
        velocity_similarity = compute_velocity_similarity(headlights_in_frames, all_association_matrices, candidate_pairs, gamma)

        # Step 13: Update weights using updated velocity and area similarity
        likelihood_matrix = get_likelihood_matrix(headlights_t, candidate_pairs)
        # print(f"likelihood_matrix = {likelihood_matrix}")
        area_similarity_matrix = get_area_similarity_matrix(headlights_t, candidate_pairs)
        # print(f"area_similarity_matrix = {area_similarity_matrix}")
        wij = update_weights(likelihood_matrix, area_similarity_matrix, velocity_similarity)
        # print(f"wij = {wij}")

        # Step 14: Solve MWIS problem using GRASP
        num_pairs = len(candidate_pairs)
        conflict_sets = create_conflict_sets(num_pairs, candidate_pairs)
        pair_weights = [(i, wij[i]) for i in range(num_pairs)]
        selected_pairs, current_pairing_energy = grasp_mwis(pair_weights, conflict_sets, candidate_pairs)
        # print(f"selected_pairs = {selected_pairs}")

        if current_pairing_energy > EGmax:
            EGmax = current_pairing_energy
            pairing_indicator = selected_pairs

        # Step 19: Compute rij (just keeping it as 0 for the first two consecutive frames)
        relation_matrix = compute_relation_similarity(headlights_t_minus_1, headlights_t, pairing_indicator,
                                                      paring_indicators_t_minus_1, gamma)

        # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteration {iter} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # print(f"+) association energy = \n {energy_function}")
        # print(f"=> FINAL ASSOCIATION MATRIX = \n {association_matrix}")
        # print(f"=> current association energy = {current_energy}")
        # print(f"+) max association energy so far = {Emax}")
        # print(f"+) paring weight = ")
        # for i in range (len(candidate_pairs)):
        #     print(f"{candidate_pairs[i]}: w = {wij[i]}")
        # print(f"=> selected_pairs = {selected_pairs}")
        # print(f"=> current paring energy = {current_pairing_energy}")
        # print(f"+) max paring energy so far = {EGmax}")
        # print(f"=> FINAL PARING INDICATOR = \n {pairing_indicator}")
        # print(f"==>>) Max Energy = {Emax}, Max Pairing Energy = {EGmax}")

    return association_matrix, pairing_indicator



# ----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------- TESTING -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# # Example usage
# headlights_t_minus_1 = np.array([Headlight(10,  20,   math.pow(  1, 2) * math.pi, math.pi/4),
#                                  Headlight(20,  20,   math.pow(  2, 2)* math.pi, math.pi/4),
#                                  Headlight(90, 100,   math.pow(3, 2)* math.pi, math.pi/4),
#                                  Headlight(50,  60,   math.pow(    1, 2) * math.pi, math.pi/4),
#                                  Headlight(60,  60,   math.pow(    1, 2) * math.pi, math.pi/4)])
#
# headlights_t         = np.array([Headlight(14,  26,   math.pow(1.2, 2) * math.pi, math.pi/4),
#                                  Headlight(90, 100,   math.pow(  3, 2)* math.pi, math.pi/4),
#                                  Headlight(24,  26,   math.pow(2.2, 2)* math.pi, math.pi/4),
#                                  Headlight(55,  66,   math.pow(  1.3, 2) * math.pi, math.pi/4),
#                                  Headlight(65,  66,   math.pow(  1.3, 2) * math.pi, math.pi/4)])
#
# headlights_t_plus_1  = np.array([Headlight(90, 100,   math.pow(  3, 2) * math.pi, math.pi/4),
#                                  Headlight(18,  32,   math.pow(1.4, 2)* math.pi, math.pi/4),
#                                  Headlight(60,  72,   math.pow(  1.6, 2) * math.pi, math.pi/4),
#                                  Headlight(28,  32,   math.pow(2.4, 2)* math.pi, math.pi/4),
#                                  Headlight(70,  72,   math.pow(  1.6, 2) * math.pi, math.pi/4)])
#
# headlights_t_plus_2  = np.array([Headlight(90, 100,   math.pow(  3, 2) * math.pi, math.pi/4),
#                                  Headlight(22,  38,   math.pow(1.6, 2)* math.pi, math.pi/4),
#                                  Headlight(32,  38,   math.pow(2.6, 2)* math.pi, math.pi/4),
#                                  Headlight(65,  78,   math.pow(  1.9, 2) * math.pi, math.pi/4),
#                                  Headlight(75,  78,   math.pow(  1.9, 2) * math.pi, math.pi/4)])
#
# headlights_in_frames = [headlights_t_minus_1, headlights_t]
# association_matrices = []
# paring_indicators_t_minus_1 = None
#
# assignment_matrix, pairing_indicator = headlight_tracking_and_pairing(headlights_in_frames, [], paring_indicators_t_minus_1)
#
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
#
# headlights_in_frames.append(headlights_t_plus_1)
# association_matrices.append(assignment_matrix)
# assignment_matrix, pairing_indicator = headlight_tracking_and_pairing(headlights_in_frames, association_matrices, pairing_indicator)
#
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
# print("--------------------------------------------------------------------------------------------------------------------------------")
#
# headlights_in_frames.append(headlights_t_plus_2)
# association_matrices.append(assignment_matrix)
# assignment_matrix, pairing_indicator = headlight_tracking_and_pairing(headlights_in_frames, association_matrices, pairing_indicator)
import pickle as pkl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool, Manager
import multiprocessing
from functools import partial

TMP_DIR = "tmp/"
FILE_FOLDER = "data/"

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pkl.load(f)
    return data.dropna()

def get_xpts(data): 
    last_xpts = data['LastXPts']
    last_xpts = list(last_xpts)
    dirs = []
    for i in range(len(last_xpts)):
        new_dir = np.array(last_xpts[i])
        new_dir = new_dir.T
        dirs.append(new_dir)
    return dirs

def get_pos(data):
    poses_np = np.zeros((len(data), 3))
    pos = list(data['RxPos'])
    for i in range(len(pos)):
        pos_np = np.array(pos[i])
        poses_np[i] = pos_np
    return poses_np

def get_dirs(xpts, poses):
    dirs = []
    for i in range(len(xpts)):
        # Vector from pose to each point in xpts[i]
        vectors = xpts[i] - poses[i]
        dirs.append(vectors/np.linalg.norm(vectors, axis=1, keepdims=True))
    return dirs
    
def line_intersection(p1, d1, p2, d2):
    """
    Calculate the closest points between two lines in 3D.
    :param p1: A point on the first line
    :param d1: Direction vector of the first line
    :param p2: A point on the second line
    :param d2: Direction vector of the second line
    :return: Closest points on both lines
    """
    A = np.array([d1, -d2]).T
    b = p2 - p1
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    return p1 + t[0] * d1, p2 + t[1] * d2

def process_dir(i, dirs, poses, intersection_points, distance_threshold):
    '''
    Find intersections between the i-th transmitter and all other transmitters
    '''
    print("Processing", i)
    for j in range(len(dirs[i])):
        d1 = dirs[i][j]
        p1 = poses[i]

        for k in range(i + 1, len(dirs)):
            for l in range(len(dirs[k])):
                d2 = dirs[k][l]
                p2 = poses[k]
                # Calculate the closest points on the lines
                closest_p1, closest_p2 = line_intersection(p1, d1, p2, d2)

                # Average the closest points to get the intersection point (approximate)
                distance = np.linalg.norm(closest_p1 - closest_p2)
                if distance < distance_threshold:
                    intersection_point = (closest_p1 + closest_p2) / 2
                    intersection_points.append(intersection_point)

def find_intersections_parallel(dirs, poses, distance_threshold=1e-10):
    '''
    Use multiprocessing to find intersections between all transmitters faster
    '''
    with Manager() as manager:
        intersection_points = manager.list()

        num_processes = multiprocessing.cpu_count()
        with Pool(processes=num_processes) as pool:
            partial_process_dir = partial(process_dir, dirs=dirs, poses=poses, intersection_points=intersection_points, distance_threshold=distance_threshold)
            pool.map(partial_process_dir, range(len(dirs)))

        return np.array(intersection_points)


def prune_intersections(intersections, min_dist=0.1):
    """
    Prune intersection points to remove any that are within min_dist cm of each other.
    :param intersections: numpy array of intersection points
    :param min_dist: minimum distance threshold in centimeters
    :return: pruned list of intersection points
    """
    # Convert min_dist from cm to meters (assuming intersections are in meters)
    min_dist_m = min_dist / 100.0

    # Use DBSCAN to cluster points with a given minimum distance threshold
    clustering = DBSCAN(eps=min_dist_m, min_samples=1).fit(intersections)

    # Get the cluster labels
    labels = clustering.labels_

    # For each cluster, take the centroid as the representative point
    unique_labels = set(labels)
    pruned_intersections = []
    for label in unique_labels:
        cluster_points = intersections[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        pruned_intersections.append(centroid)

    return np.array(pruned_intersections)

def prune_intersections_faster(intersections, min_dist=0.1):
    if not len(intersections):
        return np.array([])

    # Initialize pruned_intersections with the first point
    pruned_intersections = np.array([intersections[0]])

    for point in tqdm(intersections[1:]):
        # Compute distances from the current point to all points in pruned_intersections
        distances = np.linalg.norm(point - pruned_intersections, axis=1)
        
        # If all distances are greater than or equal to min_dist, add the point to pruned_intersections
        if np.all(distances >= min_dist):
            pruned_intersections = np.vstack([pruned_intersections, point])

    return pruned_intersections

def find_likely_transmitter(poses, dirs, pruned_intersections):
    results = []

    pruned_intersections = np.array(pruned_intersections)
    for i in tqdm(range(len(poses))):
        pose = poses[i]
        directions = dirs[i]
        
        # Calculate direction vectors from pose to each pruned intersection
        directions_from_pose = pruned_intersections - pose
        norms = np.linalg.norm(directions_from_pose, axis=1, keepdims=True)
        directions_from_pose /= norms
        
        nearest_intersections_for_pose = []
        for j in range(len(directions)):
            similarity = np.dot(directions_from_pose, directions[j])
            nearest_intersection = pruned_intersections[np.argmax(similarity)]
            nearest_intersections_for_pose.append(nearest_intersection)

        results.append(nearest_intersections_for_pose)

    return results


def predict_vtxs_given_count(train_poses, test_poses, train_vtxs, num_vtxs_per_test_tx, k=3):
    result = []
    for i in range(len(test_poses)):
        # Find the k-nearest train poses to the current test pose
        nearest_indices = np.argsort(np.linalg.norm(train_poses - test_poses[i], axis=1))[:k]
        
        candidate_vtxs = {}
        for idx in nearest_indices:
            vtxs = train_vtxs[idx]
            for vtx in vtxs:
                vtx_tuple = tuple(vtx) if isinstance(vtx, np.ndarray) else vtx
                candidate_vtxs[vtx_tuple] = candidate_vtxs.get(vtx_tuple, 0) + 1
        
        # Sort the candidate vertices by their frequency and select the top ones
        best_vtxs = sorted(candidate_vtxs, key=candidate_vtxs.get, reverse=True)
        result.append(best_vtxs[:num_vtxs_per_test_tx[i]])
    
    return result


def convert_likely_txs_to_doa(likely_txs, poses):
        doas = []
        for i in range(len(likely_txs)):
            doa = []
            for j in range(len(likely_txs[i])):
                direction = likely_txs[i][j] - poses[i]
                direction /= np.linalg.norm(direction)
                azimuth = np.arctan2(direction[1], direction[0])
                elevation = np.arcsin(direction[2])
                doa.append((azimuth, elevation))
            doas.append(doa)
        return doas


def alter_xpts(file, doas, train_split=0.8):
    # Load data
    data = load_pkl(file)
    
    # Split data into training and testing sets
    train_examples = int(len(data) * train_split)
    test_data = data[train_examples:].copy()  # Use a copy to avoid modifying the original dataframe directly
    test_poses = np.array(test_data['RxPos'])
    
    # Alter `LastXPts` in test data
    for i in range(len(test_data)):
        try:
            val = test_data['LastXPts'][train_examples + i]
        except:
            print("skipping ", i)
            continue
        
        new_doas = np.array(doas[i])
        start_pos = test_poses[i]
        new_xpts = np.zeros((len(new_doas), 3))
        
        for j in range(len(new_doas)):
            azimuth, elevation = new_doas[j]
            direction = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation)
            ])
            new_xpts[j] = start_pos + direction
        
        # Transpose new_xpts if needed
        new_xpts = new_xpts.T
        
        #data.at[train_examples + i, 'LastXPts'] = new_xpts  # Ensure it is in the correct format
        #print(train_examples + i)
        #before = data['LastXPts'][train_examples + i]
        data['LastXPts'][train_examples + i] = new_xpts
        #after = data['LastXPts'][train_examples + i]   
        #print("Before: ", before, "\nAfter: ", after)
    
    return data

def get_doas(file, overwrite_saved_values=False, display_intermediate_values=False, use_fast_prune=True, train_split=0.8):
    data = load_pkl(file)

    num_train_examples = int(len(data) * train_split)
    train_data = data[:num_train_examples]
    
    train_xpts = get_xpts(train_data)
    train_poses = get_pos(train_data)
    train_dirs = get_dirs(train_xpts, train_poses)
    file_name = file.split("/")[-1]
    print("Finding intersections")
    if (not os.path.exists(TMP_DIR + file_name[:-4] + '_all_intersections.npy')) or overwrite_saved_values:
        intersections = find_intersections_parallel(train_dirs, train_poses)
        np.save(TMP_DIR + file_name[:-4] + '_all_intersections.npy', intersections)
    
    intersections = np.load(TMP_DIR + file_name[:-4] + '_all_intersections.npy')
    print("Intersections: ", intersections.shape)

    if use_fast_prune:
        if (not os.path.exists(TMP_DIR + file_name[:-4] + '_intersections_sparse_fast.npy')) or overwrite_saved_values:
            pruned_intersections = prune_intersections_faster(intersections, min_dist=0.05)
            np.save(TMP_DIR + file_name[:-4] + '_intersections_sparse_fast.npy', pruned_intersections)
        pruned_intersections = np.load(TMP_DIR + file_name[:-4] + '_intersections_sparse_fast.npy')
    else:
        if (not os.path.exists(TMP_DIR + file_name[:-4] + '_intersections_sparse.npy')) or overwrite_saved_values:
            pruned_intersections = prune_intersections(intersections, min_dist=0.05)
            np.save(TMP_DIR + file_name[:-4] + '_intersections_sparse.npy', pruned_intersections)
        pruned_intersections = np.load(TMP_DIR + file_name[:-4] + '_intersections_sparse.npy')
    
    print("Pruned Intersections: ", pruned_intersections.shape)

    if display_intermediate_values:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pruned_intersections[:, 0], pruned_intersections[:, 1], pruned_intersections[:, 2], c='b', marker='x')
        plt.show()

    print("Finding likely transmitters")
    train_vtxs = find_likely_transmitter(train_poses, train_dirs, pruned_intersections)
    num_vtxs_per_tx = [len(vtxs) for vtxs in train_vtxs]
    num_vtxs_per_tx = np.array(num_vtxs_per_tx)
    
    print("Training MLP")
    mlp = get_mlp(train_poses, train_xpts)
    print("MLP Trained")

    # ACCESSING TEST DATA!!!!
    test_data = data[num_train_examples:]
    test_poses = get_pos(test_data)

    predicted_num_vtxs_per_test_tx = np.array(mlp.predict(test_poses))

    predicted_vtxs = predict_vtxs_given_count(train_poses, test_poses, train_vtxs, predicted_num_vtxs_per_test_tx, k=6)
    predicted_test_doas = convert_likely_txs_to_doa(predicted_vtxs, test_poses)

    return predicted_test_doas

def get_mlp(train_poses, train_xpts):
    num_xpts_per_train_tx = np.array([len(xpts) for xpts in train_xpts])
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), random_state=0)
    mlp.fit(train_poses, num_xpts_per_train_tx)
    return mlp

def ray_searching(file):
    RUN_MAIN_ANYWAY = True
    USE_FAST_PRUNE = True

    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    result_name = TMP_DIR + file[:-4] + "_doas.pkl" if not USE_FAST_PRUNE else TMP_DIR + file[:-4] + "_doas_fp.pkl"

    if (not os.path.exists(result_name)) or RUN_MAIN_ANYWAY:
        doas = get_doas(file, overwrite_saved_values=False, display_intermediate_values=False, use_fast_prune=True, train_split=0.8)
        altered_data = alter_xpts(file, doas)
        pkl.dump(altered_data, open(file[:-4] + "_searched_doa.pkl", "wb"))

if __name__ == '__main__':
    ray_searching(FILE_FOLDER + "dataset_conference_ch1_rt_image_fc.pkl")
    # ray_searching("dataset_bedroom_ch1_rt_image_fc.pkl")
    # ray_searching("dataset_office_ch1_rt_image_fc.pkl")
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import yaml
from tqdm import tqdm
from matplotlib import image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_colors(image_data, x_range, y_range):
    image_test = image_data[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    image_test_scaled = image_test.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2).fit(image_test_scaled)
    color_clusters = kmeans.cluster_centers_.astype(int)
    image_test_clustered = color_clusters[kmeans.predict(image_test_scaled)]

    fig, ax = plt.subplots(1, 2, figsize=(5, 10))
    ax[0].imshow(image_test)
    ax[1].imshow(image_test_clustered.reshape(image_test.shape))
    
    plt.savefig("color_clustering.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    return kmeans


def recolor_image(image_data, image_center, radius, colonie_color, bkg_color, color_distance_cut=20):
    fig, ax = plt.subplots(1, 2, figsize=(5, 10))
    ax[0].imshow(image_data)
    ax[0].plot(image_center[0], image_center[1], marker="o", color="white")
    circle = plt.Circle((image_center[0], image_center[1]), radius, fill=False, color="white")
    ax[0].add_patch(circle)
    ax[0].set_xlim(image_center[0]-radius, image_center[0]+radius)
    ax[0].set_ylim(image_center[1]+radius, image_center[1]-radius)

    color_distance_matrix = np.linalg.norm(image_data - colonie_color, axis=2)
    label_matrix = np.zeros(color_distance_matrix.shape)

    pixel_coord_matrix = np.mgrid[0:image_data.shape[0]:1, 0:image_data.shape[1]:1].reshape(2,-1).T
    center_distance_matrix = np.linalg.norm(pixel_coord_matrix - image_center, axis=1).reshape(color_distance_matrix.shape)

    color_distance_cut = np.linalg.norm(colonie_color - bkg_color) / 2
    image_data[color_distance_matrix > color_distance_cut] = bkg_color
    image_data[color_distance_matrix <= color_distance_cut] = colonie_color
    image_data[center_distance_matrix > radius] = bkg_color
    label_matrix[color_distance_matrix <= color_distance_cut] = 1
    label_matrix[center_distance_matrix > radius] = 0

    ax[1].imshow(label_matrix)
    plt.savefig("image_recoloring.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    return image_data, label_matrix


def radians_to_degree(angle_list):
    angle_list_degree = angle_list / np.pi * 180
    return (angle_list_degree + 360) % 360


def degree_to_radians(angle_list):
    return angle_list / 180 * np.pi


def count_angular_pieces(pixel_labels, angles_to_cut, angular_pieces_center, min_counts, max_counts):
    kmeans_best_list = []
    pixel_coord_matrix = np.mgrid[0:pixel_labels.shape[0]:1, 0:pixel_labels.shape[1]:1].reshape(2,-1).T
    pixel_center_vectors = pixel_coord_matrix - angular_pieces_center[::-1]
    pixel_center_vectors_norm = np.zeros(pixel_center_vectors.shape)
    pixel_center_vectors_norm[:,0] = pixel_center_vectors[:,0] / np.linalg.norm(pixel_center_vectors, axis=1)
    pixel_center_vectors_norm[:,1] = pixel_center_vectors[:,1] / np.linalg.norm(pixel_center_vectors, axis=1)
    vector_angle_list = radians_to_degree(np.arctan2(pixel_center_vectors_norm[:,1], -pixel_center_vectors_norm[:,0]))
    #plt.imshow(pixel_labels)
    #plt.plot(angular_pieces_center[0], angular_pieces_center[1], "o", color="red")
    #plt.show()
    fig_scores, ax_scores = plt.subplots(len(angles_to_cut), 1, figsize=(4, len(angles_to_cut)*4))
    for angle_index, angle_cut in enumerate(angles_to_cut):
        lower_bound = angle_cut
        upper_bound = angles_to_cut[(angle_index+1)%len(angles_to_cut)]
        #pixel_to_count = pixel_coord_matrix[vector_angle_list > lower_bound and vector_angle_list <= upper_bound and ]
        #print(pixel_coord_matrix.shape)
        #print(pixel_labels.shape)
        
        #angular_mask = vector_angle_list.flatten() > lower_bound  and vector_angle_list.flatten() <= upper_bound
        if lower_bound < upper_bound:
            mask_lower = vector_angle_list.flatten() > lower_bound
            mask_upper = vector_angle_list.flatten() <= upper_bound
            angular_mask = np.logical_and(mask_lower, mask_upper)
        else:
            mask_lower_one = vector_angle_list.flatten() > lower_bound
            mask_upper_one = vector_angle_list.flatten() <= 360
            mask_lower_two = vector_angle_list.flatten() > 0
            mask_upper_two = vector_angle_list.flatten() <= upper_bound
            angular_mask_one = np.logical_and(mask_lower_one, mask_upper_one)
            angular_mask_two = np.logical_and(mask_lower_two, mask_upper_two)
            angular_mask = np.logical_or(angular_mask_one, angular_mask_two)
        #print(angular_mask)
        pixel_in_piece = pixel_coord_matrix[angular_mask][pixel_labels.flatten()[angular_mask]==1]
        silhoutte_scores = np.array([])
        kmeans_list_temp = []
        progress_bar = tqdm(total=max_counts[angle_index]-min_counts[angle_index], desc=f"Angles: {lower_bound}-{upper_bound}")
        for n_clusters in range(min_counts[angle_index], max_counts[angle_index]): 
            kmeans = KMeans(n_clusters=n_clusters).fit(pixel_in_piece)
            labels = kmeans.labels_
            silhoutte_scores = np.append(silhoutte_scores, silhouette_score(pixel_in_piece, labels))
            kmeans_list_temp.append(kmeans)
            progress_bar.update(1)
        #plt.plot(range(1, max_count+1), silhoutte_scores)
        #plt.show()
        progress_bar.close()
        ax_scores.flatten()[angle_index].plot(range(min_counts[angle_index], max_counts[angle_index]), silhoutte_scores)
        ax_scores.flatten()[angle_index].plot(range(min_counts[angle_index], max_counts[angle_index])[np.argmax(silhoutte_scores)], 
                                              np.max(silhoutte_scores), marker="o", color="red")

        #plt.plot(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,0], "o", ls="", color="white", alpha=0.4)
        kmeans_best_list.append(kmeans_list_temp[np.argmax(silhoutte_scores)])
    plt.savefig("cluster_scores.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    return kmeans_best_list


def print_results(cut_angles, count_list):
    print()
    print(" "*10, "+", "-"*10, "+", "-"*10, "+")
    print(" "*10, "|", " Interval ", "|", "  Counts  ", "|")
    for i in range(len(cut_angles)):
        print(" "*10, "+", "-"*10, "+", "-"*10, "+")
        print(" "*10, "+", f"  {cut_angles[i]:3}-{cut_angles[(i+1)%len(cut_angles)]:3} ", "|", f"    {count_list[i]:3}   ", "|")
    print(" "*10, "+", "-"*10, "+", "-"*10, "+")
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", action="store_true", help="Run script without counting for parameter checking")
    parser.add_argument("--plots", "-p", action="store_true", help="Show generated plots while running")
    parser.set_defaults(test=False)
    parser.set_defaults(plots=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arguments = parse_arguments()
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    res_scale = config["resolution_scale"]
    angular_pieces_center = np.array(config["angular_pieces_center"])
    #angles_to_cut = np.array([85, 200, 330])
    angles_to_cut = np.array(config["cut_angles"])
    #max_counts = np.array([30, 250, 130])
    max_counts = np.array(config["max_count_per_angle"])
    #min_counts = np.array([15, 249, 110])
    min_counts = np.array(config["min_count_per_angle"])
    #x_range_test = np.array([500, 1000])
    x_range_test = np.array(config["x_range_test"])
    #y_range_test = np.array([500, 1000])
    y_range_test = np.array(config["y_range_test"])
    x_range_test = (x_range_test * res_scale).astype(int)
    y_range_test = (y_range_test * res_scale).astype(int)

    data = cv2.imread(config["image_file"])
    data = cv2.resize(data, dsize=(0,0), fx=res_scale, fy=res_scale)
    data_shape = data.shape
    image_center = np.array([int(data_shape[1]/2), int(data_shape[0]/2)])
    angular_pieces_center = (angular_pieces_center * res_scale).astype(int)
    circle = plt.Circle((image_center[0], image_center[1]), data_shape[0]/2, fill=False, color="white")
    plt.gca().add_patch(circle)
    plt.plot(image_center[0], image_center[1], marker="o", color="white")
    plt.plot(angular_pieces_center[0], angular_pieces_center[1], marker="o", color="white")
    for angle in angles_to_cut:
        x_coord = angular_pieces_center[0] + data_shape[0]/2 * np.sin(degree_to_radians(angle))
        y_coord = angular_pieces_center[1] - data_shape[0]/2 * np.cos(degree_to_radians(angle))
        plt.plot([angular_pieces_center[0], x_coord], [angular_pieces_center[1], y_coord], color="white")
    
    plt.imshow(data)
    #plt.plot(image_center[0], image_center[1], marker="o", color="white")
    plt.show()

    kmeans = cluster_colors(data, x_range_test, y_range_test)
    data_recolored = kmeans.cluster_centers_[kmeans.predict(data.reshape(-1, 3))].astype(int)
    colors, counts = np.unique(data_recolored, return_counts=True, axis=0)
    color_colonies = colors[np.argmin(counts)].astype(int)
    color_bkg = colors[np.argmax(counts)].astype(int)
    data_recolored, label_matrix = recolor_image(data, image_center, data_shape[0]/2, color_colonies, color_bkg)
    #plt.imshow(data_recolored)
    #plt.show()
    if not arguments.test:
        kmeans_best_list = count_angular_pieces(label_matrix, angles_to_cut, angular_pieces_center, min_counts, max_counts)
        plt.imshow(label_matrix)
        count_list = []
        for kmeans in kmeans_best_list:
            plt.plot(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,0], ls="", marker="o", color="white", alpha=0.6)
            count_list.append(len(kmeans.cluster_centers_))
        print_results(angles_to_cut, count_list)
        plt.savefig("cluster_results.pdf", bbox_inches="tight")
        plt.tight_layout()
        plt.show()

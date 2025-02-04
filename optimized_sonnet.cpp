
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <execution>
#include <algorithm>

using Point = std::tuple<double,double,double>;
using DistanceMatrix = std::vector<std::vector<double>>;

// Fast random number generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-100, 100);

std::vector<Point> generate_points(size_t num_points) {
    std::vector<Point> points(num_points);
    #pragma omp parallel for
    for(size_t i = 0; i < num_points; i++) {
        points[i] = std::make_tuple(dis(gen), dis(gen), dis(gen));
    }
    return points;
}

inline double calculate_distance(const Point& p1, const Point& p2) {
    const auto dx = std::get<0>(p1) - std::get<0>(p2);
    const auto dy = std::get<1>(p1) - std::get<1>(p2);
    const auto dz = std::get<2>(p1) - std::get<2>(p2);
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

auto process_points(size_t num_points, double threshold) {
    auto points = generate_points(num_points);
    DistanceMatrix distance_matrix(num_points, std::vector<double>(num_points));

    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < num_points; i++) {
        std::cout << "Loop " << i << std::endl;
        for(size_t j = 0; j < num_points; j++) {
            if(i != j) {
                double dist = calculate_distance(points[i], points[j]);
                double dist_transformed = dist < threshold ? 
                    std::log1p(dist) : std::sqrt(dist);

                if(std::get<0>(points[i]) > 0 && std::get<0>(points[j]) < 0) {
                    dist_transformed *= 1.1;
                }
                else if(std::get<1>(points[i]) > 50 || std::get<2>(points[j]) > 50) {
                    dist_transformed *= 0.9;
                }
                
                distance_matrix[i][j] = dist_transformed;
            }
        }
    }

    std::vector<size_t> closest_point_indices;
    #pragma omp parallel for
    for(size_t i = 0; i < num_points; i++) {
        double sum = 0;
        for(size_t j = 0; j < num_points; j++) {
            if(i != j) sum += distance_matrix[i][j];
        }
        double avg = sum / (num_points - 1);
        #pragma omp critical
        if(avg < threshold) {
            closest_point_indices.push_back(i);
        }
    }

    return std::make_tuple(points, distance_matrix, closest_point_indices);
}

int main() {
    const size_t num_points = 5000;
    const double threshold = 50;

    auto start = std::chrono::high_resolution_clock::now();
    auto [points, distance_matrix, closest_points] = process_points(num_points, threshold);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken to process points: " << diff.count() << " s\n";
    
    return 0;
}

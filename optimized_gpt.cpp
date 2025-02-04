
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;
using namespace chrono;

struct Point {
    double x, y, z;
};

double calculate_distance(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) 
              + (p1.y - p2.y) * (p1.y - p2.y) 
              + (p1.z - p2.z) * (p1.z - p2.z));
}

vector<Point> generate_points(int num_points) {
    vector<Point> points(num_points);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-100, 100);

    for (int i = 0; i < num_points; ++i) {
        points[i] = {dis(gen), dis(gen), dis(gen)};
    }
    return points;
}

void process_points(int num_points, double threshold) {
    vector<Point> points = generate_points(num_points);
    vector<vector<double>> distance_matrix(num_points, vector<double>(num_points, 0.0));

    for (int i = 0; i < num_points; ++i) {
        cout << "Loop " << i << endl;
        for (int j = 0; j < num_points; ++j) {
            if (i != j) {
                double dist = calculate_distance(points[i], points[j]);
                double dist_transformed;
                
                if (dist < threshold) {
                    dist_transformed = log1p(dist);
                } else {
                    dist_transformed = sqrt(dist);
                }

                if (points[i].x > 0 && points[j].x < 0) {
                    dist_transformed *= 1.1;
                } else if (points[i].y > 50 || points[j].z > 50) {
                    dist_transformed *= 0.9;
                }

                distance_matrix[i][j] = dist_transformed;
            }
        }
    }
    
    vector<int> closest_point_indices;
    for (int i = 0; i < num_points; ++i) {
        double sum_distances = 0;
        for (int j = 0; j < num_points; ++j) {
            if (i != j) {
                sum_distances += distance_matrix[i][j];
            }
        }
        double avg_distance = sum_distances / (num_points - 1);
        if (avg_distance < threshold) {
            closest_point_indices.push_back(i);
        }
    }

    // Printing is omitted to reduce the runtime
}

int main() {
    int num_points = 5000;
    double threshold = 50;

    auto start_time = high_resolution_clock::now();
    process_points(num_points, threshold);
    auto end_time = high_resolution_clock::now();
    
    cout << "Time taken to process points: " 
         << duration_cast<milliseconds>(end_time - start_time).count() 
         << "ms" << endl;

    return 0;
}

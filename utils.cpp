#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

// Point constructor implementations
Point::Point() : 
    danceability(0.0), 
    valence(0.0), 
    energy(0.0), 
    cluster(-1), 
    minDist(DBL_MAX) {}

Point::Point(double d, double v, double e) : 
    danceability(d), 
    valence(v), 
    energy(e), 
    cluster(-1), 
    minDist(DBL_MAX) {}

// Distance calculation implementation
double Point::distance(Point p) {
    return (p.danceability - danceability) * (p.danceability - danceability) +
           (p.valence - valence) * (p.valence - valence) + 
           (p.energy - energy) * (p.energy - energy);
}

// CSV reading implementation
std::vector<Point> readcsv() {
    std::vector<Point> points;
    std::string line;
    std::ifstream file("tracks_features.csv");

    if (!file.is_open()) {
        throw std::runtime_error("Could not open tracks_features.csv");
    }

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;
        bool inQuotes = false;
        std::string current;

        // Handle quoted fields with commas
        for (char c : line) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                row.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        row.push_back(current);

        // Check if we have enough columns
        if (row.size() >= 19) {
            try {
                double danceability = std::stod(row[9]);   // Column 9
                double valence = std::stod(row[18]);       // Column 18
                double energy = std::stod(row[10]);       // Column 10
                points.push_back(Point(danceability, valence, energy));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Skipping malformed line - " << e.what() << "\n";
                continue;
            }
        }
    }
    
    if (points.empty()) {
        throw std::runtime_error("No valid data points loaded from CSV");
    }

    return points;
}

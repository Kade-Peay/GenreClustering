#pragma once
#include <vector>
#include <string>
#include <cfloat>

struct Point {
    double danceability;
    double valence;
    double energy;
    int cluster;
    double minDist;

    Point();
    Point(double d, double v, double e);
    double distance(Point p);
};

constexpr double convergenceDelta = 1e-10;

std::vector<Point> readcsv(std::string filename);

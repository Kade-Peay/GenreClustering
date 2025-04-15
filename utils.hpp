#pragma once
#include <vector>
#include <string>

struct Point
{
    double x, y;
    int cluster;
    double minDist;

    Point();
    Point(double x, double y);
    double distance(Point p);
};

std::vector<Point> readcsv(std::string filename);
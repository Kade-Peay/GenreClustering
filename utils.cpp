#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cfloat>

Point::Point() : x(0.0), y(0.0), cluster(-1), minDist(DBL_MAX) {}

Point::Point(double x, double y) : x(x), y(y), cluster(-1), minDist(DBL_MAX) {}

double Point::distance(Point p)
{
    return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
}

std::vector<Point> readcsv()
{
    std::vector<Point> points;
    std::string line;
    std::ifstream file("tracks_features.csv");

    std::getline(file, line); // Skip header

    while (std::getline(file, line))
    {
        std::vector<std::string> row;
        bool inQuotes = false;
        std::string current;

        for (char c : line)
        {
            if (c == '"') inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes)
            {
                row.push_back(current);
                current.clear();
            }
            else current += c;
        }
        row.push_back(current);

        if (row.size() >= 11)
        {
            try
            {
                size_t pos;
                double x = std::stod(row[9], &pos);
                if (pos != row[9].length()) continue;
                double y = std::stod(row[10], &pos);
                if (pos != row[10].length()) continue;

                points.push_back(Point(x, y));
            }
            catch (const std::exception& e)
            {
                std::cerr << "Skipping line due to parse error: " << line << "\n";
            }
        }
    }
    return points;
}
CC = gcc
CFLAGS = -O3 -Wall -Wextra -fopenmp
LDFLAGS = -lm
TARGET = kmeans_omp
SRC = kmeans_omp.c utils.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

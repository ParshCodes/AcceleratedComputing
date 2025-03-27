CC = gcc
CFLAGS = -O2 -Wall -fopenmp
SERIAL_SRC = serial_LL.c
PARALLEL_SRC = parallel_LL.c
SERIAL_EXEC = serial
PARALLEL_EXEC = parallel

all: $(SERIAL_EXEC) $(PARALLEL_EXEC)

$(SERIAL_EXEC): $(SERIAL_SRC)
	$(CC) $(CFLAGS) -o $@ $<

$(PARALLEL_EXEC): $(PARALLEL_SRC)
	$(CC) $(CFLAGS) -o $@ $<

run_serial:
	./$(SERIAL_EXEC) > serial_results.csv

run_parallel:
	./$(PARALLEL_EXEC) > parallel_results.csv


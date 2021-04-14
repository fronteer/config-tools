CC=g++

CFLAGS := -std=c++11 -I/opt/rocm/include -I./ -g 

PROGRAMS :=  generate_configs  reorder_configs_bwd  reorder_configs_fwd

all: $(PROGRAMS)

# Step

generate_configs: generate_configs.o
	$(CC) -o $@ $< 

reorder_configs_bwd: reorder_configs_bwd.o
	$(CC) -o $@ $< 

reorder_configs_fwd: reorder_configs_fwd.o
	$(CC) -o $@ $< 

produce_header: produce_header.o
	$(CC) -o $@ $< 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $< 


clean:
	rm -f *.o $(PROGRAMS)


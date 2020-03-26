# makefile for single directory c++ projects

# target & files
TARGET := prefix_sum
DBG_SUFFIX := _dbg
TARGET_DEBUG := $(addsuffix $(DBG_SUFFIX), $(TARGET)) # main_dbg
# MAIN_SRC := $(addsuffix .cpp, $(TARGET))
SRC := $(basename $(wildcard *.c*))

# macros
CC := $(if $(shell command -v mpiicpc), mpiicpc, mpic++)
CCFLAG := -O3 #-std=c++11
DBGFLAG := -DDEBUG # -g -Wall -std=c++11
CCOBJFLAG := $(CCFLAG) -c

# targets
$(TARGET): $(addsuffix .o, $(SRC))
	$(CC) $(CCFLAG) -o $@ $^

%.o: %.c*
	$(CC) $(CCOBJFLAG) -o $@ $<

$(TARGET_DEBUG): $(addsuffix $(DBG_SUFFIX).o, $(SRC))
	$(CC) $(CCFLAG) $(DBGFLAG) -o $(TARGET) $^

%$(DBG_SUFFIX).o: %.c*
	$(CC) $(CCOBJFLAG) $(DBGFLAG) -o $@ $<

#default rule
defualt: all

# phony
.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: clean $(TARGET_DEBUG)

.PHONY: run
run:
	@echo "Running $(TARGET) with $(N) cores ..."
	@-mpiexec -n $(N) ./$(TARGET)

.PHONY: run_default
run_default:
	@echo "Running $(TARGET) with 3 cores using input.txt ..."
	@-mpiexec -n 4 ./$(TARGET)

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(TARGET) $(TARGET_DEBUG)

.PHONY: clear
swipe:
	rm -f *.o

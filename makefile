# makefile for single directory c++ projects

# target & files
TARGET := main 
DBG_SUFFIX := _dbg
TARGET_DEBUG := $(addsuffix $(DBG_SUFFIX), $(TARGET)) # main_dbg
# MAIN_SRC := $(addsuffix .cpp, $(TARGET))
SRC := $(basename $(wildcard *.c*))

# macros
CC := mpicc
CCFLAG := -O3 #-std=c++11
DBGFLAG := -g -Wall # -std=c++11
CCOBJFLAG := $(CCFLAG) -c

# targets
$(TARGET): $(addsuffix .o, $(SRC))
	$(CC) $(CCFLAG) -o $@ $^

%.o: %.c*
	$(CC) $(CCOBJFLAG) -o $@ $<

$(TARGET_DEBUG): $(addsuffix $(DBG_SUFFIX).o, $(SRC))
	$(CC) $(CCFLAG) $(DBGFLAG) -o $@ $^

%$(DBG_SUFFIX).o: %.c*
	$(CC) $(CCOBJFLAG) $(DBGFLAG) -o $@ $<

#default rule
defualt: all

# phony
.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(TARGET) $(TARGET_DEBUG)

.PHONY: clear
swipe:
	rm -f *.o

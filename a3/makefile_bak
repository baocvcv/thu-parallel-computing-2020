default:matvectmul
CC = mpiicc
CFLAGS = -O3
CLIBS = 
matvectmul:prog3_56.o
	$(CC) $(CLIBS) prog3_56.o -o matvectmul
prog3_56.o:prog3_56.cpp
	$(CC) $(CFLAGS) -c prog3_56.cpp -o prog3_56.o
clean:
	rm -rf ./*.o matvectmul

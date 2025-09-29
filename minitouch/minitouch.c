#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <time.h>

#include <libevdev.h>

#define MAX_CONTACTS 10
#define MAX_TIME_POINT 100000
#define MAX_OPT 100000

int contacts[MAX_CONTACTS];
int fd = -1;
int tracking_id = 0;
struct libevdev* evdev = NULL;

typedef struct{
    int time;
    int cursor;
} TimePoint;

typedef struct{
    short ty;
    short contact;
    short x;
    short y;
} Opt;


TimePoint time_list[MAX_TIME_POINT];
Opt opt_list[MAX_OPT];
int time_top, opt_top;
int first_beat_time;
long long start_time;
int delta_time;
char pref_file[256];

long long shift_time = 0;
int shift_time_cnt = 0;

static int consider_device(const char* devpath){
    if ((fd = open(devpath, O_RDWR)) < 0){
        perror("open");
        fprintf(stderr, "Unable to open device %s for inspection\n", devpath);
        goto mismatch;
    }
    if (libevdev_new_from_fd(fd, &evdev) < 0){
        fprintf(stderr, "Note: device %s is not supported by libevdev\n", devpath);
        goto mismatch;
    }
    return 0;
mismatch:
    libevdev_free(evdev);
    if (fd >= 0) close(fd);
    return 1;
}
static int write_event(uint16_t type, uint16_t code, int32_t value){
    struct input_event event = {{0, 0}, type, code, value};
    ssize_t length = (ssize_t) sizeof(event);
    return write(fd, &event, length) - length;
}
static int next_tracking_id(){
    return tracking_id < INT_MAX ? (tracking_id += 1) : (tracking_id = 1);
}
static int commit() {
    write_event(EV_SYN, SYN_REPORT, 0);
    return 1;
}
static int touch_down(int contact, int x, int y){
    contacts[contact]= next_tracking_id();
    //fprintf(stderr, "d %d %d %d %d\n", contact, x, y, pressure);
    write_event(EV_ABS, ABS_MT_SLOT, contact);
    write_event(EV_ABS, ABS_MT_TRACKING_ID, contacts[contact]);

    write_event(EV_ABS, ABS_MT_TOUCH_MAJOR, 0x00000006);
    write_event(EV_ABS, ABS_MT_WIDTH_MAJOR, 0x00000004);
    write_event(EV_ABS, ABS_MT_PRESSURE, 10);

    write_event(EV_ABS, ABS_MT_POSITION_X, x);
    write_event(EV_ABS, ABS_MT_POSITION_Y, y);

    return 1;
}
static int touch_move(int contact, int x, int y){
    //fprintf(stderr, "m %d %d %d %d\n", contact, x, y, pressure);
    write_event(EV_ABS, ABS_MT_SLOT, contact);
    
    write_event(EV_ABS, ABS_MT_TOUCH_MAJOR, 0x00000006);
    write_event(EV_ABS, ABS_MT_WIDTH_MAJOR, 0x00000004);
    write_event(EV_ABS, ABS_MT_PRESSURE, 10);

    write_event(EV_ABS, ABS_MT_POSITION_X, x);
    write_event(EV_ABS, ABS_MT_POSITION_Y, y);
    return 1;
}
static int touch_up(int contact){
    //fprintf(stderr, "u %d\n", contact);
    contacts[contact]= 0;
    write_event(EV_ABS, ABS_MT_SLOT, contact);
    write_event(EV_ABS, ABS_MT_TRACKING_ID, -1);
    return 1;
}
static int touch_panic_reset_all(){
    int contact, found_any = 0;
    for (contact = 0; contact < MAX_CONTACTS; ++contact){
        if (contacts[contact]) found_any = 1;
        touch_up(contact);
    }
    return found_any ? commit() : 0;
}
static void start_player(){
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    #define ADD_THRESHOLD 110000ll
    long long 
		cur_time = now.tv_sec*1000000ll+now.tv_nsec/1000,
		miss_time = cur_time-start_time;
//		add_time = 
//			miss_time > ADD_THRESHOLD ? miss_time-ADD_THRESHOLD :
//			miss_time < ADD_THRESHOLD ? miss_time-ADD_THRESHOLD : 0;
//    
    fprintf(stderr, 
		"start_time:%lld cur_time:%lld miss_time:%lld within:%lld\n", 
        start_time, cur_time, miss_time,
		start_time+delta_time-cur_time);
    start_time += delta_time;
    
    int opt_cursor = 0, time_cursor = 0, t, p;
    short ty, contact, x, y;
    while(opt_cursor < opt_top){
        clock_gettime(CLOCK_REALTIME, &now);
        t = now.tv_sec*1000000ll+now.tv_nsec/1000-start_time;
        TimePoint *tp;
        while (time_cursor < time_top && t > (tp = &(time_list[time_cursor]))->time){
            p = tp->cursor;
            while(opt_cursor < p){
                ty = opt_list[opt_cursor].ty;
                contact = opt_list[opt_cursor].contact;
                x = opt_list[opt_cursor].x;
                y = opt_list[opt_cursor].y;
                switch(ty){
                case 0:
                    touch_down(contact, x, y);
                    break;
                case 1:
                    touch_move(contact, x, y);
                    break;
                case 2:
                    touch_up(contact);
                    break;
                case 3:
                    commit();
                    break;
                default:
                    break;
                }
                ++opt_cursor;
            }
            ++time_cursor;
        }
    }

    fprintf(stderr, "Finished playing.\n");
}
static void parse_pref_input(char* buffer){
    char* cursor;
    int t, contact, x, y, pressure;

    cursor = (char*) buffer;
    cursor += 1;
    
    switch (buffer[0]) {
        case 'b': // COMMIT
            first_beat_time = strtol(cursor, &cursor, 10);
            break;
        case 't': // COMMIT
            t = strtol(cursor, &cursor, 10);
            time_list[time_top].time = t-first_beat_time;
            if (time_top) time_list[time_top-1].cursor = opt_top;
            time_top += 1;
            break;
        case 'd': // TOUCH DOWN
            opt_list[opt_top].ty = 0;
            opt_list[opt_top].contact = strtol(cursor, &cursor, 10);
            opt_list[opt_top].x = strtol(cursor, &cursor, 10);
            opt_list[opt_top].y = strtol(cursor, &cursor, 10);
            ++opt_top;
            break;
        case 'm': // TOUCH MOVE
            opt_list[opt_top].ty = 1;
            opt_list[opt_top].contact = strtol(cursor, &cursor, 10);
            opt_list[opt_top].x = strtol(cursor, &cursor, 10);
            opt_list[opt_top].y = strtol(cursor, &cursor, 10);
            ++opt_top;
            break;
        case 'u': // TOUCH UP
            opt_list[opt_top].ty = 2;
            opt_list[opt_top].contact = strtol(cursor, &cursor, 10);
            ++opt_top;
            break;
        case 'c':
            opt_list[opt_top].ty = 3;
            ++opt_top;
            break;
        default:
            break;
    }
}
static void read_pref(FILE* input){
    setvbuf(input, NULL, _IOLBF, 1024);

    time_top = opt_top = 0;
    char read_buffer[80];
    //int cnt = 0;
    while (fgets(read_buffer, sizeof(read_buffer), input) != NULL){
    	//printf("cnt:%d\n", cnt++);
        read_buffer[strcspn(read_buffer, "\r\n")] = 0;
        parse_pref_input(read_buffer);
    }
   printf("time_top:%d opt_top:%d\n", time_top, opt_top);
    time_list[time_top-1].cursor = opt_top;
}
static void parse_input(char* buffer){
    char* cursor;
    int contact, x, y, pressure;

    cursor = (char*) buffer;
    cursor += 1;

    struct timespec now;
    switch (buffer[0]) {
        case 'c': // COMMIT
            commit();
            break;
        case 'r': // RESET
            touch_panic_reset_all();
            break;
        case 'd': // TOUCH DOWN
            contact = strtol(cursor, &cursor, 10);
            x = strtol(cursor, &cursor, 10);
            y = strtol(cursor, &cursor, 10);
            if (contacts[contact]){
                fprintf(stderr, "Failed 'touch down' on contact %d %d %d\n", contact, x, y);
                break;
            }
            touch_down(contact, x, y);
            break;
        case 'm': // TOUCH MOVE
            contact = strtol(cursor, &cursor, 10);
            x = strtol(cursor, &cursor, 10);
            y = strtol(cursor, &cursor, 10);
            if (!contacts[contact]){
                fprintf(stderr, "Failed 'touch move' on contact %d %d %d\n", contact, x, y);
                break;
            }
            touch_move(contact, x, y);
            break;
        case 'u': // TOUCH UP
            contact = strtol(cursor, &cursor, 10);
            if (!contacts[contact]){
                fprintf(stderr, "Failed 'touch up' on contact %d\n", contact);
                break;
            }
            touch_up(contact);
            break;
        case 't': // used to align time
            clock_gettime(CLOCK_REALTIME, &now);
            long long cur_time = now.tv_sec*1000000ll+now.tv_nsec/1000;
            long long t = strtoll(cursor, &cursor, 10);
            if (t < 0){
                shift_time /= shift_time_cnt;
                fprintf(stderr, "shift_time:%lld\n", shift_time);
            }
            else{
                shift_time += t-cur_time;
                shift_time_cnt += 1;
            }
            break;
        case 'f':
		    FILE *pref_input = fopen(pref_file, "r");
		    if (pref_input == NULL){
		        fprintf(stderr, "Unable to open '%s': %s\n", pref_file, strerror(errno));
		        exit(EXIT_FAILURE);
		    }
		    read_pref(pref_input);
		    fclose(pref_input);
		    break;
        case 's':
            start_time = strtoll(cursor, &cursor, 10)+shift_time;
            delta_time = strtol(cursor, &cursor, 10);
            start_player();
            break;
        case 'q':
			libevdev_free(evdev);
            close(fd);
            exit(EXIT_SUCCESS);
			break; 
        default:
            break;
    }
}
static void io_handler(FILE* input, FILE* output){
    setvbuf(input, NULL, _IOLBF, 1024);
    setvbuf(output, NULL, _IOLBF, 1024);

    char read_buffer[80];

    while (fgets(read_buffer, sizeof(read_buffer), input) != NULL){
        read_buffer[strcspn(read_buffer, "\r\n")] = 0;
        parse_input(read_buffer);
    }
}
int main(int argc, char* argv[]){
    const char* devpath = "/dev/input/event1";
    if (argc < 2){
        fprintf(stderr, "Unable to find a prefabricated file.");
        return EXIT_FAILURE;
    }
    strcpy(pref_file, argv[1]);
    FILE *pref_input = fopen(pref_file, "r");
    if (pref_input == NULL){
        fprintf(stderr, "Unable to open '%s': %s\n", pref_file, strerror(errno));
        exit(EXIT_FAILURE);
    }
    read_pref(pref_input);
    fclose(pref_input);

    consider_device(devpath);

    if (evdev == NULL){
        fprintf(stderr, "Unable to find a suitable touch device\n");
        return EXIT_FAILURE;
    }

    FILE* input = NULL, *output = NULL;

    input = stdin;
    fprintf(stderr, "Reading from STDIN\n");

    output = stderr;
    io_handler(input, output);

    fclose(input);
    fclose(output);
    exit(EXIT_SUCCESS);

    libevdev_free(evdev);
    close(fd);

    return EXIT_SUCCESS;
}

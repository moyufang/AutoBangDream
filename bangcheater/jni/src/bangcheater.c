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

#include "Log.c"
#include "controller.c"
#include <libevdev.h>

#define MAX_CONTACTS 10
#define MAX_TIME_POINT 100000
#define MAX_OPT 100000

#define PORT 12345
#define BUFFER_SIZE 1024

int WIDTH = 1280, HEIGHT = 720; 
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
char commands_file_path[256];
long long cur_time = 0;
long long start_time;
long long shift_time = 0;
long long correction_time;
long long dilation_time = 1000000ll;
long long diff_time = 0;
int first_beat_time;
int predict_time;

static int consider_device(const char* devpath){
    if ((fd = open(devpath, O_RDWR)) < 0){
        perror("open");
        LogE("Unable to open device %s for inspection\n", devpath);
        goto mismatch;
    }
    if (libevdev_new_from_fd(fd, &evdev) < 0){
        LogE("Note: device %s is not supported by libevdev\n", devpath);
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
    //LogD("d %d %d %d %d\n", contact, x, y, pressure);

    write_event(EV_ABS, ABS_MT_SLOT, contact);
    write_event(EV_ABS, ABS_MT_TRACKING_ID, contacts[contact]);

    write_event(EV_ABS, ABS_MT_TOUCH_MAJOR, 0x00000006);
    write_event(EV_ABS, ABS_MT_WIDTH_MAJOR, 0x00000004);
    write_event(EV_ABS, ABS_MT_PRESSURE, 10);

    write_event(EV_ABS, ABS_MT_POSITION_X, HEIGHT-1-y);
    write_event(EV_ABS, ABS_MT_POSITION_Y, x);

    return 1;
}
static int touch_move(int contact, int x, int y){
    //LogD("m %d %d %d %d\n", contact, x, y, pressure);
    write_event(EV_ABS, ABS_MT_SLOT, contact);
    
    write_event(EV_ABS, ABS_MT_TOUCH_MAJOR, 0x00000006);
    write_event(EV_ABS, ABS_MT_WIDTH_MAJOR, 0x00000004);
    write_event(EV_ABS, ABS_MT_PRESSURE, 10);

    write_event(EV_ABS, ABS_MT_POSITION_X, HEIGHT-1-y);
    write_event(EV_ABS, ABS_MT_POSITION_Y, x);
    return 1;
}
static int touch_up(int contact){
    //LogD("u %d\n", contact);
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
static void caliboration(){
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  long long cur_time = now.tv_sec*1000000ll+now.tv_nsec/1000;
  long long here_first_beat_time = start_time + predict_time + shift_time + correction_time;
  diff_time = here_first_beat_time-cur_time;
  LogI(
    "Calibration para:\n\tcur_time:%lld h_first_beat_time:%lld\n\tdiff:%lld\n", 
    cur_time, here_first_beat_time, here_first_beat_time-cur_time
  );

}
static void start_player(int right_now){
  for(int i = 0; i < MAX_CONTACTS; ++i) touch_up(i); commit();
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  long long cur_time = now.tv_sec*1000000ll+now.tv_nsec/1000;
  long long here_first_beat_time = start_time + predict_time + shift_time + correction_time;
  if (right_now) here_first_beat_time = cur_time;
  LogI(
    "Calibration para:\n\tcur_time:%lld h_first_beat_time:%lld\n\tdiff:%lld\n", 
    cur_time, here_first_beat_time, here_first_beat_time-cur_time
  );
  
  long long hsh = 0, hsh_base = 131, hsh_mod = 998244353ll;
  for(int i = 0; i < opt_top; ++i){
    hsh = ((hsh*hsh_base)+opt_list[i].ty)%hsh_mod;
    hsh = ((hsh*hsh_base)+opt_list[i].contact)%hsh_mod;
    hsh = ((hsh*hsh_base)+opt_list[i].x)%hsh_mod;
    hsh = ((hsh*hsh_base)+opt_list[i].y)%hsh_mod;
  }
  for(int i = 0; i < time_top; ++i){
    hsh = ((hsh*hsh_base)+time_list[i].cursor)%hsh_mod;
    hsh = ((hsh*hsh_base)+time_list[i].time)%hsh_mod;
  }
  LogD("COMMANDS HASH: %lld\n", hsh);

  int opt_cursor = 0, time_cursor = 0, t, p;
  short ty, contact, x, y;
  while(opt_cursor < opt_top){
    clock_gettime(CLOCK_REALTIME, &now);
    t = now.tv_sec*1000000ll+now.tv_nsec/1000-here_first_beat_time;
    TimePoint *tp;
    while(
      time_cursor < time_top && 
      t > ((tp = &(time_list[time_cursor]))->time*dilation_time)/1000000ll
    ){
      // LogD("run cmds t:%d time_cursor:%d p:%d opt_cursor:%d\n", t, time_cursor, tp->cursor, opt_cursor);
      p = tp->cursor;
      while(opt_cursor < p){
        ty = opt_list[opt_cursor].ty;
        contact = opt_list[opt_cursor].contact;
        x = opt_list[opt_cursor].x;
        y = opt_list[opt_cursor].y;
        switch(ty){
          case 0: touch_down(contact, x, y);
          // LogD("d %d %d %d\n", contact, x, y);
          break;
          case 1: touch_move(contact, x, y);
          // LogD("m %d %d %d\n", contact, x, y);
          break;
          case 2: touch_up(contact); 
          // LogD("u %d\n", contact);
          break;
          default:break;
        }
        ++opt_cursor;
      }
      commit();
      // LogD("c\n");
      ++time_cursor;
    }
  }

    fprintf(stderr, "Finished playing.\n");
}
static void read_commands(FILE* input){
  LogI("Read commands from '%s'\n", commands_file_path);
  
  time_top = opt_top = 0;
  char buffer[64];
  while (fscanf(input, "%s", buffer) != EOF){
    int t, contact, x, y, pressure;
    switch (buffer[0]) {
      case 'b': // FIRST BEAT TIME
        fscanf(input, "%d", &first_beat_time);
        break;
      case 't': // SYN TIME
        fscanf(input, "%d", &t);
        time_list[time_top].time = t-first_beat_time;
        time_list[time_top].time = time_list[time_top].time;
        // LogD("t:%d -> %d\n", t, time_list[time_top].time);
        time_list[time_top++].cursor = opt_top;
        break;
      case 'd': // TOUCH DOWN
        opt_list[opt_top].ty = 0;
        fscanf(input, "%hd %hd %hd",
          &opt_list[opt_top].contact,
          &opt_list[opt_top].x,
          &opt_list[opt_top].y);
        // LogD("d %d %d %d\n",opt_list[opt_top].contact,opt_list[opt_top].x,opt_list[opt_top].y);
        ++opt_top;
        break;
      case 'm': // TOUCH MOVE
        opt_list[opt_top].ty = 1;
        fscanf(input, "%hd %hd %hd",
          &opt_list[opt_top].contact,
          &opt_list[opt_top].x,
          &opt_list[opt_top].y);
        // LogD("m %d %d %d\n",opt_list[opt_top].contact,opt_list[opt_top].x,opt_list[opt_top].y);
        ++opt_top;
        break;
      case 'u': // TOUCH UP
        opt_list[opt_top].ty = 2;
        fscanf(input, "%hd", &opt_list[opt_top].contact);
        // LogD("u %d\n", opt_list[opt_top].contact);
        ++opt_top;
        break;
      default:
        break;
    }
  }
  time_list[time_top-1].cursor = opt_top;
}
static int parse_input(const char* buffer){
  FILE *commands_input;
  char* cursor;
  int contact, x, y, pressure;

  cursor = (char*) buffer;
  cursor += 1;

  // LogR("Parse input:%s\n", buffer);

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
          LogE("Failed 'touch down' on contact %d %d %d\n", contact, x, y);
          break;
      }
      touch_down(contact, x, y);
      break;
    case 'm': // TOUCH MOVE
      contact = strtol(cursor, &cursor, 10);
      x = strtol(cursor, &cursor, 10);
      y = strtol(cursor, &cursor, 10);
      if (!contacts[contact]){
          LogE("Failed 'touch move' on contact %d %d %d\n", contact, x, y);
          break;
      }
      touch_move(contact, x, y);
      break;
    case 'u': // TOUCH UP
      contact = strtol(cursor, &cursor, 10);
      if (!contacts[contact]){
          LogE("Failed 'touch up' on contact %d\n", contact);
          break;
      }
      touch_up(contact);
      break;
    case 't': // used to align time
      clock_gettime(CLOCK_REALTIME, &now);
      long long rcv_time = now.tv_sec*1000000ll+now.tv_nsec/1000;
      long long snd_time = strtoll(cursor, &cursor, 10);
      correction_time = strtoll(cursor, &cursor, 10);
      dilation_time = strtoll(cursor, &cursor, 10);
      shift_time = rcv_time - snd_time;
      start_time = snd_time;
      LogR("Synchronize time:\nsnd_time:%lld rcv_time:%lld\nshift_time:%lld correction_time:%d\n", snd_time, rcv_time, shift_time, correction_time);
      break;
    case 's':
      predict_time = strtol(cursor, &cursor, 10);
      start_player(0);
      break;
    case 'C':
      predict_time = strtol(cursor, &cursor, 10);
      caliboration();
      return CONTROLLER_CALIBORATION;
      break;
    case 'f':
      commands_input = fopen(commands_file_path, "r");
      if (commands_input == NULL){
        LogE("Unable to open '%s': %s\n", commands_file_path, strerror(errno));
        exit(EXIT_FAILURE);
      }
      read_commands(commands_input);
      fclose(commands_input);
      break;
    case 'n':
      start_player(1);
      break;
    case 'q':
      return CONTROLLER_QUIT_CONNECTION;
      break;
    case 'k':
    case 'e':
      return CONTROLLER_EXIT;
      break;
    case 'p':
      return CONTROLLER_READY;
      break;
    default:
      break;
  }
  return 0;
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
static int tcp_io(const char *buffer, char *response){
  int ret_code = parse_input(buffer);
  switch (ret_code){
    case CONTROLLER_READY: return CONTROLLER_READY;
    case CONTROLLER_EXIT: return CONTROLLER_EXIT;
    case CONTROLLER_CALIBORATION:
      sprintf(response, "%lld", diff_time);
      return CONTROLLER_CALIBORATION;
    default: response[0] = 0; return 0;
  }
  return 0;
}

int main(int argc, char* argv[]){

  //<============ Get input device ============>
  const char* devpath = "/dev/input/event4";
  LogI("Depend device: %s\n", devpath);
  consider_device(devpath);

  if (evdev == NULL){
    LogE("Unable to find a suitable touch device\n");
    return EXIT_FAILURE;
  }

  //<============ Get commands ============>
    
  if (argc < 2){
    LogE("Unable to find a commands file.\n");
    return EXIT_FAILURE;
  }
  strcpy(commands_file_path, argv[1]);
  // FILE *commands_input = fopen(commands_file_path, "r");
  // if (commands_input == NULL){
  //   LogE( "Unable to open '%s': %s\n", commands_file_path, strerror(errno));
  //   exit(EXIT_FAILURE);
  // }
  // read_commands(commands_input);
  
  // fclose(commands_input);

  //<============ Switch Mode ============>

  int mode;
  if (argc >= 3){
    //adb
    if (argv[2][0] == '-' && argv[2][1] == 'i') mode = 1; 
    //tcp
    else if (argv[2][0] == '-' && argv[2][1] == 't') mode = 2;
  }
  else mode = 1;

  //<============ ADB ============>
  if (mode == 1){
    LogI("Reading from STDIN\n");
    
    FILE* input = NULL, *output = NULL;
    input = stdin;
    output = stderr;
    io_handler(input, output);


    fclose(input);
    fclose(output);
  }
  //<============ TCP ============>
  else if(mode == 2){
    LogI("Reading from TCP PORT:%d\n", PORT);

    struct LowLatencyController clr;
    init_controller(&clr, PORT, BUFFER_SIZE, tcp_io);
    launch_controller(&clr);
  }

  LogI("Kill bangcheater.\n");
  libevdev_free(evdev);
  close(fd);

  return EXIT_SUCCESS;
}

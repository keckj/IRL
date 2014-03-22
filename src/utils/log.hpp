
#ifndef LOG_H
#define LOG_H

#include "log4cpp/Category.hh"

extern log4cpp::Category& log_console;
extern log4cpp::Category& log_file;

void initLogs();

#endif /* end of include guard: LOG_H */

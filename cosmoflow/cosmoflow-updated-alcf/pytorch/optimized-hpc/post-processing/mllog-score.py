#!/usr/bin/env python3

import fileinput
import re
import json

###########################################################################
#
# Generate a single run result score from an individual mlperf log file
#
###########################################################################

class FilePos:
    def __init__(self):
        self.state='before'

    def run_start(self, json_struct):
        self.start_time = json_struct['time_ms']
        if not (self.state == 'before' or self.state == 'run_stop'):
            print(999999999.9)
        self.state = 'run_start'

    def run_stop(self, json_struct):
        stop_time = json_struct['time_ms']
        result_success = json_struct['metadata']['status']
        if result_success != 'success' or self.state != 'run_start':
            print(999999999.9)
        else:
            print((float(stop_time)-float(self.start_time))/1000.0)
        self.state = 'run_stop'

    def file_end(self):
        if self.state == 'before':
            # no job ever started, so no time
            print()
        elif self.state != 'run_stop':
            # job started but never converged and never_stopped
            print(999999999.9)
        self.state = 'before'

    def dispatch_json(self, json_struct):
        if json_struct['key'] == 'run_start':
            self.run_start(json_struct)
        elif json_struct['key'] == 'run_stop':
            self.run_stop(json_struct)

    def process_line(self, line):
        json_match = re.match(r'.*:::MLLOG\s+' + r'(.*)', line)
        if json_match:
            json_string = json_match.group(1)
            json_struct = json.loads(json_string)
            self.dispatch_json(json_struct)

def main():
    file_pos = FilePos()
    for line in fileinput.input(openhook=fileinput.hook_encoded("utf-8", errors="ignore")):
        file_pos.process_line(line)
    file_pos.file_end()

if __name__=='__main__':
    main()
